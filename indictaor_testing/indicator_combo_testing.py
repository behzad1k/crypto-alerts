import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta


class MultiTimeframeBacktester:
  def __init__(self, csv_path: str, data_slice: int = None):
    """
    Initialize backtester with 1h CSV data

    Args:
        csv_path: Path to CSV file
        data_slice: If provided, only use last N rows (e.g., 5000 for faster testing)
    """
    self.csv_path = csv_path
    self.data_slice = data_slice
    self.timeframes = {
      '1h': 60,
      '2h': 120,
      '4h': 240,
      '6h': 360,
      '8h': 480,
      '12h': 720,
      '1d': 1440
    }

    # Individual indicators and their combinations
    self.individual_indicators = [
      'MACD_crossover', 'MACD_position',
      'Momentum_5', 'Momentum_10',
      'RSI_oversold', 'RSI_overbought', 'RSI_midline',
      'STOCH_D_oversold', 'STOCH_D_overbought', 'STOCH_D_midline',
      'WILLR_overbought', 'WILLR_oversold', 'WILLR_midline'
    ]

    self.combination_indicators = [
      'MACD_RSI_bullish', 'MACD_RSI_bearish',
      'MACD_MOM_bullish', 'MACD_MOM_bearish',
      'RSI_STOCH_oversold', 'RSI_STOCH_overbought',
      'RSI_WILLR_oversold', 'RSI_WILLR_overbought',
      'MOM_5_10_bullish', 'MOM_5_10_bearish',
      'STOCH_WILLR_oversold', 'STOCH_WILLR_overbought'
    ]

    self.all_indicators = self.individual_indicators + self.combination_indicators

  def load_data(self) -> pd.DataFrame:
    """Load 1h data from CSV"""
    df = pd.read_csv(self.csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Slice data if requested
    if self.data_slice:
      df = df.tail(self.data_slice).reset_index(drop=True)
      print(f"Using last {self.data_slice} rows for faster processing")

    return df

  def derive_timeframe(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Derive higher timeframe from 1h data"""
    if target_tf == '1h':
      return df.copy()

    tf_minutes = self.timeframes[target_tf]
    periods = tf_minutes // 60  # How many 1h bars per target timeframe

    result = []
    for i in range(0, len(df), periods):
      chunk = df.iloc[i:i + periods]
      if len(chunk) == 0:
        continue

      agg = {
        'Date': chunk.iloc[-1]['Date'],
        'Symbol': chunk.iloc[0]['Symbol'],
        'Open': chunk.iloc[0]['Open'],
        'High': chunk['High'].max(),
        'Low': chunk['Low'].min(),
        'Close': chunk.iloc[-1]['Close'],
        'Volume': chunk['Volume'].sum()
      }
      result.append(agg)

    return pd.DataFrame(result)

  def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    data = df.copy()

    if len(data) < 50:
      return data

    # MACD
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic %D
    lowest_low = data['Low'].rolling(window=14).min()
    highest_high = data['High'].rolling(window=14).max()
    k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    data['STOCH_K'] = k_percent.rolling(window=3).mean()
    data['STOCH_D'] = data['STOCH_K'].rolling(window=3).mean()

    # Williams %R
    data['WILLR'] = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))

    return data

  def generate_signals(self, data: pd.DataFrame, idx: int) -> Dict[str, str]:
    """Generate signals for a specific row"""
    if idx < 1 or idx >= len(data):
      return {}

    signals = {}
    curr = data.iloc[idx]
    prev = data.iloc[idx - 1]

    # MACD Signals
    if not pd.isna(curr['MACD']) and not pd.isna(curr['MACD_signal']):
      if curr['MACD'] > curr['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        signals['MACD_crossover'] = 'BUY'
      elif curr['MACD'] < curr['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        signals['MACD_crossover'] = 'SELL'

      if curr['MACD'] > 0:
        signals['MACD_position'] = 'BUY'
      else:
        signals['MACD_position'] = 'SELL'

    # Momentum Signals
    if not pd.isna(curr['Momentum_5']):
      if curr['Momentum_5'] > 0.02:
        signals['Momentum_5'] = 'BUY'
      elif curr['Momentum_5'] < -0.02:
        signals['Momentum_5'] = 'SELL'

    if not pd.isna(curr['Momentum_10']):
      if curr['Momentum_10'] > 0.03:
        signals['Momentum_10'] = 'BUY'
      elif curr['Momentum_10'] < -0.03:
        signals['Momentum_10'] = 'SELL'

    # RSI Signals
    if not pd.isna(curr['RSI']):
      if curr['RSI'] < 30:
        signals['RSI_oversold'] = 'BUY'
      elif curr['RSI'] > 70:
        signals['RSI_overbought'] = 'SELL'

      if curr['RSI'] > 50 and prev['RSI'] <= 50:
        signals['RSI_midline'] = 'BUY'
      elif curr['RSI'] < 50 and prev['RSI'] >= 50:
        signals['RSI_midline'] = 'SELL'

    # Stochastic %D Signals
    if not pd.isna(curr['STOCH_D']):
      if curr['STOCH_D'] < 20:
        signals['STOCH_D_oversold'] = 'BUY'
      elif curr['STOCH_D'] > 80:
        signals['STOCH_D_overbought'] = 'SELL'

      if curr['STOCH_D'] > 50 and prev['STOCH_D'] <= 50:
        signals['STOCH_D_midline'] = 'BUY'
      elif curr['STOCH_D'] < 50 and prev['STOCH_D'] >= 50:
        signals['STOCH_D_midline'] = 'SELL'

    # Williams %R Signals
    if not pd.isna(curr['WILLR']):
      if curr['WILLR'] > -20:
        signals['WILLR_overbought'] = 'SELL'
      elif curr['WILLR'] < -80:
        signals['WILLR_oversold'] = 'BUY'

      if curr['WILLR'] > -50 and prev['WILLR'] <= -50:
        signals['WILLR_midline'] = 'BUY'
      elif curr['WILLR'] < -50 and prev['WILLR'] >= -50:
        signals['WILLR_midline'] = 'SELL'

    # Combination Signals
    if (not pd.isna(curr['MACD']) and not pd.isna(curr['MACD_signal']) and
      not pd.isna(curr['RSI'])):
      if curr['MACD'] > curr['MACD_signal'] and curr['RSI'] < 30:
        signals['MACD_RSI_bullish'] = 'BUY'
      elif curr['MACD'] < curr['MACD_signal'] and curr['RSI'] > 70:
        signals['MACD_RSI_bearish'] = 'SELL'

    if (not pd.isna(curr['MACD']) and not pd.isna(curr['MACD_signal']) and
      not pd.isna(curr['Momentum_10'])):
      if curr['MACD'] > curr['MACD_signal'] and curr['Momentum_10'] > 0.02:
        signals['MACD_MOM_bullish'] = 'BUY'
      elif curr['MACD'] < curr['MACD_signal'] and curr['Momentum_10'] < -0.02:
        signals['MACD_MOM_bearish'] = 'SELL'

    if not pd.isna(curr['RSI']) and not pd.isna(curr['STOCH_D']):
      if curr['RSI'] < 30 and curr['STOCH_D'] < 20:
        signals['RSI_STOCH_oversold'] = 'BUY'
      elif curr['RSI'] > 70 and curr['STOCH_D'] > 80:
        signals['RSI_STOCH_overbought'] = 'SELL'

    if not pd.isna(curr['RSI']) and not pd.isna(curr['WILLR']):
      if curr['RSI'] < 30 and curr['WILLR'] < -80:
        signals['RSI_WILLR_oversold'] = 'BUY'
      elif curr['RSI'] > 70 and curr['WILLR'] > -20:
        signals['RSI_WILLR_overbought'] = 'SELL'

    if not pd.isna(curr['Momentum_5']) and not pd.isna(curr['Momentum_10']):
      if curr['Momentum_5'] > 0.02 and curr['Momentum_10'] > 0.03:
        signals['MOM_5_10_bullish'] = 'BUY'
      elif curr['Momentum_5'] < -0.02 and curr['Momentum_10'] < -0.03:
        signals['MOM_5_10_bearish'] = 'SELL'

    if not pd.isna(curr['STOCH_D']) and not pd.isna(curr['WILLR']):
      if curr['STOCH_D'] < 20 and curr['WILLR'] < -80:
        signals['STOCH_WILLR_oversold'] = 'BUY'
      elif curr['STOCH_D'] > 80 and curr['WILLR'] > -20:
        signals['STOCH_WILLR_overbought'] = 'SELL'

    return signals

  def backtest_combination(self, df_1h: pd.DataFrame, combo: List[Tuple[str, str]],
                           holding_periods: List[int] = [4, 8, 12, 24],
                           tf_data_cache: Dict = None, tf_signals_cache: Dict = None) -> Dict:
    """
    Backtest a 3-indicator combination across timeframes
    combo format: [(timeframe, indicator), (timeframe, indicator), (timeframe, indicator)]
    Uses cached data if provided for performance
    """
    # Use cache if provided, otherwise calculate
    if tf_data_cache is None or tf_signals_cache is None:
      tf_data = {}
      tf_signals = {}

      for tf in self.timeframes.keys():
        df_tf = self.derive_timeframe(df_1h, tf)
        df_tf = self.calculate_indicators(df_tf)
        tf_data[tf] = df_tf

        # Generate signals for all rows
        signals_list = []
        for i in range(len(df_tf)):
          signals_list.append(self.generate_signals(df_tf, i))
        tf_signals[tf] = signals_list
    else:
      tf_data = tf_data_cache
      tf_signals = tf_signals_cache

    # Track trades
    trades = []

    # Scan through 1h data (our base timeframe for entry/exit)
    for i in range(50, len(df_1h) - max(holding_periods)):
      current_date = df_1h.iloc[i]['Date']

      # Check if all 3 indicators align on their respective timeframes
      signals_match = []
      signal_types = []

      for tf, indicator in combo:
        # Find corresponding index in this timeframe
        df_tf = tf_data[tf]
        tf_idx = None

        for idx, row in df_tf.iterrows():
          if row['Date'] <= current_date:
            tf_idx = idx
          else:
            break

        if tf_idx is None or tf_idx >= len(tf_signals[tf]):
          signals_match.append(False)
          continue

        signals = tf_signals[tf][tf_idx]
        if indicator in signals:
          signals_match.append(True)
          signal_types.append(signals[indicator])
        else:
          signals_match.append(False)

      # All 3 must match and be the same type (all BUY or all SELL)
      if all(signals_match) and len(set(signal_types)) == 1:
        signal_type = signal_types[0]
        entry_price = df_1h.iloc[i]['Close']
        entry_date = current_date

        # Test different holding periods
        for hold_periods in holding_periods:
          if i + hold_periods >= len(df_1h):
            continue

          exit_idx = i + hold_periods
          exit_price = df_1h.iloc[exit_idx]['Close']
          exit_date = df_1h.iloc[exit_idx]['Date']

          if signal_type == 'BUY':
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
          else:  # SELL
            profit_pct = ((entry_price - exit_price) / entry_price) * 100

          trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'signal': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'holding_periods': hold_periods,
            'win': profit_pct > 0
          })

    if len(trades) == 0:
      return {
        'combination': combo,
        'total_trades': 0,
        'win_rate': 0,
        'avg_profit': 0,
        'total_profit': 0,
        'max_drawdown': 0
      }

    # Calculate statistics
    wins = sum(1 for t in trades if t['win'])
    win_rate = (wins / len(trades)) * 100
    avg_profit = np.mean([t['profit_pct'] for t in trades])
    total_profit = sum([t['profit_pct'] for t in trades])

    # Calculate max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
      cumulative += t['profit_pct']
      if cumulative > peak:
        peak = cumulative
      dd = peak - cumulative
      if dd > max_dd:
        max_dd = dd

    return {
      'combination': combo,
      'combination_str': ' + '.join([f"[{tf}] {ind}" for tf, ind in combo]),
      'total_trades': len(trades),
      'win_rate': win_rate,
      'avg_profit': avg_profit,
      'total_profit': total_profit,
      'max_drawdown': max_dd,
      'trades': trades
    }

  def run_comprehensive_backtest(self, max_combinations: int = None, random_sample: bool = False):
    """
    Run backtest on all possible 3-indicator combinations

    Args:
        max_combinations: Limit number of combinations to test
        random_sample: If True, randomly sample combinations instead of sequential
    """
    print("Loading data...")
    df_1h = self.load_data()
    print(f"Loaded {len(df_1h)} 1h candles")

    # Pre-calculate all timeframe data and signals (HUGE performance boost)
    print("Pre-calculating indicators for all timeframes...")
    tf_data_cache = {}
    tf_signals_cache = {}

    for tf in self.timeframes.keys():
      print(f"  Processing {tf} timeframe...")
      df_tf = self.derive_timeframe(df_1h, tf)
      df_tf = self.calculate_indicators(df_tf)
      tf_data_cache[tf] = df_tf

      # Generate signals for all rows
      signals_list = []
      for i in range(len(df_tf)):
        signals_list.append(self.generate_signals(df_tf, i))
      tf_signals_cache[tf] = signals_list

    print("Pre-calculation complete!")

    # Generate all possible combinations
    print("\nGenerating indicator combinations...")
    all_combos = []

    for tf1 in self.timeframes.keys():
      for ind1 in self.all_indicators:
        for tf2 in self.timeframes.keys():
          for ind2 in self.all_indicators:
            for tf3 in self.timeframes.keys():
              for ind3 in self.all_indicators:
                combo = [(tf1, ind1), (tf2, ind2), (tf3, ind3)]
                all_combos.append(combo)

    total_combos = len(all_combos)
    print(f"Total possible combinations: {total_combos:,}")

    if max_combinations:
      if random_sample:
        import random
        all_combos = random.sample(all_combos, min(max_combinations, len(all_combos)))
        print(f"Randomly sampling {len(all_combos):,} combinations")
      else:
        all_combos = all_combos[:max_combinations]
        print(f"Testing first {len(all_combos):,} combinations")

    # Run backtests
    print("\nRunning backtests...")
    results = []
    import time
    start_time = time.time()

    for i, combo in enumerate(all_combos):
      if i % 100 == 0 and i > 0:
        elapsed = time.time() - start_time
        rate = i / elapsed
        remaining = (len(all_combos) - i) / rate
        print(f"Progress: {i:,}/{len(all_combos):,} ({i / len(all_combos) * 100:.1f}%) | "
              f"Rate: {rate:.1f}/sec | ETA: {remaining / 60:.1f} min | "
              f"Found: {len(results)} valid")

      result = self.backtest_combination(df_1h, combo,
                                         tf_data_cache=tf_data_cache,
                                         tf_signals_cache=tf_signals_cache)
      if result['total_trades'] >= 10:  # Only keep combinations with at least 10 trades
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nBacktest completed in {elapsed / 60:.1f} minutes")
    print(f"Average rate: {len(all_combos) / elapsed:.1f} combinations/second")

    # Sort by win rate, then by total trades
    results.sort(key=lambda x: (x['win_rate'], x['total_trades']), reverse=True)

    return results

  def print_top_results(self, results: List[Dict], top_n: int = 50):
    """Print top N results in formatted output"""
    print(f"\n{'=' * 120}")
    print(f"TOP {top_n} INDICATOR COMBINATIONS")
    print(f"{'=' * 120}")
    print(f"{'Combination':<100} {'Win%':<8} {'Trades':<8} {'Avg%':<8}")
    print(f"{'-' * 120}")

    for i, result in enumerate(results[:top_n], 1):
      combo_str = result['combination_str']
      win_rate = result['win_rate']
      trades = result['total_trades']
      avg_profit = result['avg_profit']

      # Format infinity symbol for edge cases
      avg_str = f"{avg_profit:.1f}" if abs(avg_profit) < 1000 else "∞"

      print(f"{combo_str:<100} {win_rate:<8.1f} {trades:<8} {avg_str:<8}")

    print(f"{'=' * 120}\n")

    # Print summary statistics
    if results:
      print(f"SUMMARY STATISTICS:")
      print(f"  Total valid combinations: {len(results)}")
      print(f"  Best win rate: {results[0]['win_rate']:.1f}%")
      print(f"  Avg win rate (top 50): {np.mean([r['win_rate'] for r in results[:50]]):.1f}%")
      print(f"  Avg trades (top 50): {np.mean([r['total_trades'] for r in results[:50]]):.0f}")
      print()

  def save_results(self, results: List[Dict], filename: str = 'backtest_results.json'):
    """Save results to JSON file"""
    # Remove trades detail to keep file size manageable
    save_data = []
    for r in results:
      save_r = {k: v for k, v in r.items() if k != 'trades'}
      save_data.append(save_r)

    with open(filename, 'w') as f:
      json.dump(save_data, f, indent=2)

    print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
  # RECOMMENDED: Start with quick test on small data slice
  print("=== QUICK TEST MODE (Recommended for first run) ===")
  print("This will test 500 random combinations on last 5000 rows (~5-10 minutes)\n")

  backtester = MultiTimeframeBacktester('./datasets/Binance_TRXUSDT_1h (1).csv', data_slice=5000)
  results = backtester.run_comprehensive_backtest(
    max_combinations=1000,
    random_sample=True
  )
  backtester.print_top_results(results, top_n=20)
  backtester.save_results(results, 'quick_test_results.json')

  # Uncomment below for more thorough tests:

  # MEDIUM TEST: Last 10,000 rows, 2000 combinations (~30-45 minutes)
  # print("\n=== MEDIUM TEST MODE ===")
  # backtester = MultiTimeframeBacktester('your_data.csv', data_slice=10000)
  # results = backtester.run_comprehensive_backtest(
  #     max_combinations=2000,
  #     random_sample=True
  # )
  # backtester.print_top_results(results, top_n=50)
  # backtester.save_results(results, 'medium_test_results.json')

  # LARGE TEST: Last 20,000 rows, 5000 combinations (~2-3 hours)
  # print("\n=== LARGE TEST MODE ===")
  # backtester = MultiTimeframeBacktester('your_data.csv', data_slice=20000)
  # results = backtester.run_comprehensive_backtest(
  #     max_combinations=5000,
  #     random_sample=True
  # )
  # backtester.print_top_results(results, top_n=100)
  # backtester.save_results(results, 'large_test_results.json')

  # FULL TEST: All data, 10000+ combinations (MANY HOURS!)
  # Only run this if you're confident and have time
  print("\n=== FULL TEST MODE ===")
  backtester = MultiTimeframeBacktester('./datasets/Binance_BTCUSDT_1h (1).csv')
  results = backtester.run_comprehensive_backtest(
      max_combinations=10000,
      random_sample=True
  )
  backtester.print_top_results(results, top_n=100)
  backtester.save_results(results, 'full_test_results.json')

  print(f"\nTotal combinations with 10+ trades: {len(results)}")
  print("Done! Check the JSON file for complete results.")