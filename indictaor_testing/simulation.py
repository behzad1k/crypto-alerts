import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
  """Load multi-coin hourly data and prepare for analysis"""
  df = pd.read_csv(file_path)

  # Standardize column names
  column_mapping = {
    'timestamp': 'Date', 'Timestamp': 'Date', 'date': 'Date',
    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    'volume': 'Volume', 'symbol': 'Symbol'
  }

  for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
      df = df.rename(columns={old_name: new_name})

  # Handle timestamp conversion
  try:
    df['Date'] = pd.to_datetime(df['Date'])
  except:
    try:
      df['Date'] = pd.to_datetime(df['Date'], unit='s')
    except:
      df['Date'] = pd.to_datetime(df['Date'], unit='ms')

  df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

  return df


def prepare_coin_data(df, symbol):
  """Prepare data for a specific coin"""
  coin_data = df[df['Symbol'] == symbol].copy()
  coin_data = coin_data.set_index('Date')
  coin_data = coin_data.sort_index()

  # Remove Symbol column as it's now redundant
  if 'Symbol' in coin_data.columns:
    coin_data = coin_data.drop('Symbol', axis=1)

  return coin_data


def create_timeframes(df_1h):
  """Create multiple timeframes from 1h data"""
  timeframes = {
    '1h': df_1h.copy(),
    '2h': df_1h.resample('2H').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(),
    '4h': df_1h.resample('4H').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(),
    '6h': df_1h.resample('6H').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(),
    '8h': df_1h.resample('8H').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(),
    '12h': df_1h.resample('12H').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(),
    '1d': df_1h.resample('1D').agg({
      'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
  }

  return timeframes


def calculate_all_indicators(df):
  """Calculate all technical indicators"""
  data = df.copy()

  # RSI
  delta = data['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  data['RSI'] = 100 - (100 / (1 + rs))

  # MACD
  ema_12 = data['Close'].ewm(span=12).mean()
  ema_26 = data['Close'].ewm(span=26).mean()
  data['MACD'] = ema_12 - ema_26
  data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

  # Momentum
  data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
  data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1

  # Williams %R
  low_min = data['Low'].rolling(window=14).min()
  high_max = data['High'].rolling(window=14).max()
  data['WILLR'] = -100 * ((high_max - data['Close']) / (high_max - low_min))

  # Stochastic D
  stoch_k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
  data['STOCH_D'] = stoch_k.rolling(window=3).mean()

  return data


def generate_signals(data, indicators):
  """Generate trading signals for specified indicators"""
  signals = pd.DataFrame(index=data.index)

  for indicator in indicators:
    if indicator == 'RSI':
      signals[f'{indicator}_signal'] = np.where(
        data[indicator] < 30, 1,  # Oversold - buy
        np.where(data[indicator] > 70, -1, 0)  # Overbought - sell
      )

    elif indicator == 'MACD':
      macd_prev = data['MACD'].shift(1)
      macd_signal_prev = data['MACD_signal'].shift(1)

      bullish_cross = (data['MACD'] > data['MACD_signal']) & (macd_prev <= macd_signal_prev)
      bearish_cross = (data['MACD'] < data['MACD_signal']) & (macd_prev >= macd_signal_prev)

      signals[f'{indicator}_signal'] = np.where(bullish_cross, 1, np.where(bearish_cross, -1, 0))

    elif indicator == 'Momentum_5':
      signals[f'{indicator}_signal'] = np.where(
        data[indicator] > 0.01, 1,  # Positive momentum - buy
        np.where(data[indicator] < -0.01, -1, 0)  # Negative momentum - sell
      )

    elif indicator == 'Momentum_10':
      signals[f'{indicator}_signal'] = np.where(
        data[indicator] > 0.015, 1,  # Positive momentum - buy
        np.where(data[indicator] < -0.015, -1, 0)  # Negative momentum - sell
      )

    elif indicator == 'WILLR':
      signals[f'{indicator}_signal'] = np.where(
        data[indicator] < -80, 1,  # Oversold - buy
        np.where(data[indicator] > -20, -1, 0)  # Overbought - sell
      )

    elif indicator == 'STOCH_D':
      signals[f'{indicator}_signal'] = np.where(
        data[indicator] < 20, 1,  # Oversold - buy
        np.where(data[indicator] > 80, -1, 0)  # Overbought - sell
      )

  return signals


def combine_signals(signals, combination_indicators):
  """Combine multiple indicator signals using majority voting"""
  if len(combination_indicators) == 1:
    return signals[f'{combination_indicators[0]}_signal']

  # Get signals for the combination
  combo_signals = signals[[f'{ind}_signal' for ind in combination_indicators]]

  # Majority voting
  buy_votes = (combo_signals == 1).sum(axis=1)
  sell_votes = (combo_signals == -1).sum(axis=1)

  # Require majority agreement
  threshold = len(combination_indicators) / 2

  combined_signal = np.where(
    buy_votes > threshold, 1,
    np.where(sell_votes > threshold, -1, 0)
  )

  return pd.Series(combined_signal, index=signals.index)


class TradingSimulation:
  def __init__(self, initial_capital=1000, risk_per_trade=0.02, stop_loss_pct=0.03, take_profit_pct=0.06):
    self.initial_capital = initial_capital
    self.capital = initial_capital
    self.position = 0  # 0 = no position, 1 = long, -1 = short
    self.entry_price = 0
    self.shares = 0
    self.risk_per_trade = risk_per_trade
    self.stop_loss_pct = stop_loss_pct
    self.take_profit_pct = take_profit_pct

    # Tracking
    self.trades = []
    self.portfolio_values = []
    self.max_drawdown = 0
    self.peak_value = initial_capital

  def calculate_position_size(self, current_price):
    """Calculate position size based on risk management"""
    risk_amount = self.capital * self.risk_per_trade
    position_size = risk_amount / (current_price * self.stop_loss_pct)

    # Don't use more than 95% of capital
    max_position_value = self.capital * 0.95
    max_position_size = max_position_value / current_price

    return min(position_size, max_position_size)

  def execute_trade(self, signal, current_price, timestamp):
    """Execute trade based on signal"""
    # Check for exit conditions first
    if self.position != 0:
      # Stop loss check
      if self.position == 1:  # Long position
        if current_price <= self.entry_price * (1 - self.stop_loss_pct):
          self.close_position(current_price, timestamp, 'stop_loss')
        elif current_price >= self.entry_price * (1 + self.take_profit_pct):
          self.close_position(current_price, timestamp, 'take_profit')
        elif signal == -1:  # Signal reversal
          self.close_position(current_price, timestamp, 'signal_exit')

      elif self.position == -1:  # Short position
        if current_price >= self.entry_price * (1 + self.stop_loss_pct):
          self.close_position(current_price, timestamp, 'stop_loss')
        elif current_price <= self.entry_price * (1 - self.take_profit_pct):
          self.close_position(current_price, timestamp, 'take_profit')
        elif signal == 1:  # Signal reversal
          self.close_position(current_price, timestamp, 'signal_exit')

    # Open new position
    if self.position == 0 and signal != 0:
      position_size = self.calculate_position_size(current_price)

      if signal == 1:  # Buy signal
        cost = position_size * current_price
        if cost <= self.capital:
          self.shares = position_size
          self.capital -= cost
          self.position = 1
          self.entry_price = current_price

          self.trades.append({
            'timestamp': timestamp,
            'action': 'buy',
            'price': current_price,
            'shares': position_size,
            'cost': cost
          })

      elif signal == -1:  # Sell signal (short)
        # For simplicity, implement short as cash position
        # In real trading, this would require margin
        proceeds = position_size * current_price
        self.shares = -position_size  # Negative shares for short
        self.capital += proceeds
        self.position = -1
        self.entry_price = current_price

        self.trades.append({
          'timestamp': timestamp,
          'action': 'short',
          'price': current_price,
          'shares': -position_size,
          'proceeds': proceeds
        })

  def close_position(self, current_price, timestamp, reason):
    """Close current position"""
    if self.position == 1:  # Close long
      proceeds = self.shares * current_price
      pnl = proceeds - (self.shares * self.entry_price)
      self.capital += proceeds

    elif self.position == -1:  # Close short
      cost = abs(self.shares) * current_price
      pnl = (abs(self.shares) * self.entry_price) - cost
      self.capital -= cost

    self.trades.append({
      'timestamp': timestamp,
      'action': 'close',
      'price': current_price,
      'shares': self.shares,
      'pnl': pnl,
      'reason': reason
    })

    self.shares = 0
    self.position = 0
    self.entry_price = 0

  def update_portfolio_value(self, current_price, timestamp):
    """Update portfolio value tracking"""
    if self.position == 1:
      portfolio_value = self.capital + (self.shares * current_price)
    elif self.position == -1:
      portfolio_value = self.capital - (abs(self.shares) * (current_price - self.entry_price))
    else:
      portfolio_value = self.capital

    self.portfolio_values.append({
      'timestamp': timestamp,
      'portfolio_value': portfolio_value,
      'price': current_price
    })

    # Track drawdown
    if portfolio_value > self.peak_value:
      self.peak_value = portfolio_value

    drawdown = (self.peak_value - portfolio_value) / self.peak_value
    if drawdown > self.max_drawdown:
      self.max_drawdown = drawdown

  def get_final_stats(self):
    """Calculate final performance statistics"""
    if not self.portfolio_values:
      return {}

    final_value = self.portfolio_values[-1]['portfolio_value']
    total_return = (final_value / self.initial_capital - 1) * 100

    # Calculate Sharpe ratio (simplified)
    if len(self.portfolio_values) > 1:
      returns = []
      for i in range(1, len(self.portfolio_values)):
        prev_val = self.portfolio_values[i - 1]['portfolio_value']
        curr_val = self.portfolio_values[i]['portfolio_value']
        returns.append((curr_val / prev_val - 1))

      if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
      else:
        sharpe_ratio = 0
    else:
      sharpe_ratio = 0

    # Trading statistics
    if self.trades:
      trades_df = pd.DataFrame(self.trades)
      close_trades = trades_df[trades_df['action'] == 'close']

      if not close_trades.empty and 'pnl' in close_trades.columns:
        winning_trades = close_trades[close_trades['pnl'] > 0]
        win_rate = len(winning_trades) / len(close_trades) * 100

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = close_trades[close_trades['pnl'] < 0]
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and \
                                                                                         losing_trades[
                                                                                           'pnl'].sum() != 0 else float(
          'inf')
      else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    else:
      win_rate = 0
      avg_win = 0
      avg_loss = 0
      profit_factor = 0

    return {
      'final_value': final_value,
      'total_return': total_return,
      'max_drawdown': self.max_drawdown * 100,
      'sharpe_ratio': sharpe_ratio,
      'total_trades': len([t for t in self.trades if t['action'] == 'close']),
      'win_rate': win_rate,
      'avg_win': avg_win,
      'avg_loss': avg_loss,
      'profit_factor': profit_factor
    }


def run_comprehensive_simulation(file_path, initial_capital=1000, coins_to_test=None):
  """Run simulation across all indicators, combinations, timeframes, and coins"""

  print("Loading multi-coin data...")
  df_all = load_and_prepare_data(file_path)

  # Get available coins
  available_coins = df_all['Symbol'].unique()
  print(f"Found {len(available_coins)} coins: {', '.join(available_coins)}")

  # Filter coins if specified
  if coins_to_test:
    available_coins = [coin for coin in available_coins if coin in coins_to_test]
    print(f"Testing selected coins: {', '.join(available_coins)}")

  # Define indicators and combinations
  indicators = ['RSI', 'MACD', 'Momentum_5', 'Momentum_10', 'WILLR', 'STOCH_D']

  all_combinations = []
  for r in range(1, len(indicators) + 1):
    all_combinations.extend(list(combinations(indicators, r)))

  print(f"Testing {len(all_combinations)} indicator combinations across 7 timeframes...")
  print(f"Total scenarios per coin: {len(all_combinations) * 7}")
  print(f"Total scenarios across all coins: {len(all_combinations) * 7 * len(available_coins)}")

  all_results = []
  coin_summaries = []

  for coin_idx, coin in enumerate(available_coins, 1):
    print(f"\n{'=' * 80}")
    print(f"PROCESSING COIN {coin_idx}/{len(available_coins)}: {coin}")
    print(f"{'=' * 80}")

    # Prepare coin-specific data
    coin_data = prepare_coin_data(df_all, coin)

    if len(coin_data) < 1000:  # Need sufficient data
      print(f"Skipping {coin}: insufficient data ({len(coin_data)} records)")
      continue

    # Create timeframes for this coin
    timeframes = create_timeframes(coin_data)

    coin_results = []
    scenario_count = 0

    for timeframe_name, data in timeframes.items():
      # Calculate indicators for this timeframe
      try:
        data_with_indicators = calculate_all_indicators(data)

        # Skip first 50 periods for indicator stability
        start_idx = min(50, len(data_with_indicators) // 10)
        data_subset = data_with_indicators.iloc[start_idx:].copy()

        if len(data_subset) < 100:
          continue

        for combination in all_combinations:
          scenario_count += 1

          if scenario_count % 50 == 0:
            progress = scenario_count / (len(all_combinations) * len(timeframes))
            print(f"  {coin} Progress: {scenario_count}/{len(all_combinations) * len(timeframes)} "
                  f"({progress * 100:.1f}%)")

          try:
            # Generate signals for this combination
            signals = generate_signals(data_subset, combination)
            combined_signal = combine_signals(signals, combination)

            # Run simulation
            sim = TradingSimulation(initial_capital=initial_capital)

            for i, (timestamp, row) in enumerate(data_subset.iterrows()):
              current_price = row['Close']
              signal = combined_signal.iloc[i] if not pd.isna(combined_signal.iloc[i]) else 0

              sim.execute_trade(signal, current_price, timestamp)
              sim.update_portfolio_value(current_price, timestamp)

            # Close any remaining position
            if sim.position != 0:
              final_price = data_subset['Close'].iloc[-1]
              final_timestamp = data_subset.index[-1]
              sim.close_position(final_price, final_timestamp, 'end_of_data')

            # Get statistics
            stats = sim.get_final_stats()

            # Calculate buy & hold benchmark for this period
            start_price = data_subset['Close'].iloc[0]
            end_price = data_subset['Close'].iloc[-1]
            buy_hold_return = (end_price / start_price - 1) * 100

            # Store results with coin information
            result = {
              'coin': coin,
              'timeframe': timeframe_name,
              'indicators': list(combination),
              'indicator_count': len(combination),
              'combination_name': ' + '.join(combination),
              'buy_hold_return': buy_hold_return,
              'outperformance': stats['total_return'] - buy_hold_return,
              **stats
            }

            coin_results.append(result)

          except Exception as e:
            continue

      except Exception as e:
        print(f"  Error processing {coin} - {timeframe_name}: {e}")
        continue

    # Summarize this coin's results
    if coin_results:
      coin_df = pd.DataFrame(coin_results)

      coin_summary = {
        'coin': coin,
        'total_strategies_tested': len(coin_df),
        'profitable_strategies': len(coin_df[coin_df['total_return'] > 0]),
        'strategies_beating_hold': len(coin_df[coin_df['outperformance'] > 0]),
        'best_strategy_return': coin_df['total_return'].max(),
        'best_strategy_name': coin_df.loc[coin_df['total_return'].idxmax(), 'combination_name'],
        'best_strategy_timeframe': coin_df.loc[coin_df['total_return'].idxmax(), 'timeframe'],
        'avg_return': coin_df['total_return'].mean(),
        'avg_outperformance': coin_df['outperformance'].mean(),
        'best_sharpe': coin_df['sharpe_ratio'].max(),
        'avg_max_drawdown': coin_df['max_drawdown'].mean(),
        'data_points': len(coin_data)
      }

      coin_summaries.append(coin_summary)
      all_results.extend(coin_results)

      print(f"  {coin} Results: {len(coin_df)} strategies tested, "
            f"best return: {coin_summary['best_strategy_return']:.1f}%, "
            f"profitable: {coin_summary['profitable_strategies']}")

  return pd.DataFrame(all_results), pd.DataFrame(coin_summaries)


def analyze_simulation_results(results_df, coin_summaries_df):
  """Analyze and rank simulation results with coin profiling"""

  if results_df.empty:
    print("No results to analyze!")
    return

  print("\n" + "=" * 100)
  print("COMPREHENSIVE MULTI-COIN SIMULATION RESULTS ANALYSIS")
  print("=" * 100)

  # Overall statistics
  print(f"Total scenarios tested: {len(results_df)}")
  print(f"Coins analyzed: {len(coin_summaries_df)}")
  print(f"Scenarios with positive returns: {len(results_df[results_df['total_return'] > 0])}")
  print(f"Scenarios beating buy & hold: {len(results_df[results_df['outperformance'] > 0])}")

  # COIN PROFILING SECTION
  print(f"\n{'=' * 100}")
  print("COIN PROFILING - WHICH COINS RESPOND BEST TO TECHNICAL ANALYSIS")
  print(f"{'=' * 100}")

  # Rank coins by strategy effectiveness
  coin_rankings = coin_summaries_df.sort_values('best_strategy_return', ascending=False)

  print(f"\nCOIN RANKINGS BY BEST STRATEGY PERFORMANCE:")
  print("-" * 100)
  print(
    f"{'Rank':<5} {'Coin':<12} {'Best Return':<12} {'Avg Return':<12} {'Profitable%':<12} {'Beat Hold%':<12} {'Best Strategy'}")
  print("-" * 100)

  for i, (_, coin) in enumerate(coin_rankings.iterrows(), 1):
    profitable_pct = coin['profitable_strategies'] / coin['total_strategies_tested'] * 100
    beat_hold_pct = coin['strategies_beating_hold'] / coin['total_strategies_tested'] * 100

    print(f"{i:<5} {coin['coin']:<12} {coin['best_strategy_return']:>10.1f}% "
          f"{coin['avg_return']:>10.1f}% {profitable_pct:>10.1f}% "
          f"{beat_hold_pct:>10.1f}% {coin['best_strategy_name'][:25]}")

  # Coin-specific insights
  print(f"\nCOIN-SPECIFIC INSIGHTS:")
  print("-" * 80)

  # Most consistent performers
  coin_summaries_df['consistency_score'] = (
    coin_summaries_df['profitable_strategies'] / coin_summaries_df['total_strategies_tested'] *
    coin_summaries_df['avg_return'] / 100
  )

  most_consistent = coin_summaries_df.nlargest(5, 'consistency_score')
  print(f"Most Consistent Performers (profitable strategies × avg return):")
  for _, coin in most_consistent.iterrows():
    print(f"  {coin['coin']:<12}: {coin['consistency_score']:.3f} "
          f"({coin['profitable_strategies']}/{coin['total_strategies_tested']} profitable, "
          f"{coin['avg_return']:.1f}% avg return)")

  # Best technical analysis candidates
  best_technical = coin_summaries_df.nlargest(5, 'strategies_beating_hold')
  print(f"\nBest Technical Analysis Candidates (strategies beating buy & hold):")
  for _, coin in best_technical.iterrows():
    beat_pct = coin['strategies_beating_hold'] / coin['total_strategies_tested'] * 100
    print(f"  {coin['coin']:<12}: {coin['strategies_beating_hold']}/{coin['total_strategies_tested']} "
          f"strategies ({beat_pct:.1f}%) beat buy & hold")

  # OVERALL STRATEGY ANALYSIS
  print(f"\n{'=' * 100}")
  print("OVERALL BEST STRATEGIES ACROSS ALL COINS")
  print(f"{'=' * 100}")

  # Top strategies by total return
  print(f"\nTOP 15 STRATEGIES BY TOTAL RETURN:")
  print("-" * 100)
  print(f"{'Rank':<5} {'Coin':<12} {'Timeframe':<10} {'Strategy':<30} {'Return':<10} {'Outperform':<12} {'Drawdown'}")
  print("-" * 100)

  top_returns = results_df.nlargest(15, 'total_return')
  for i, (_, row) in enumerate(top_returns.iterrows(), 1):
    print(f"{i:<5} {row['coin']:<12} {row['timeframe']:<10} {row['combination_name'][:29]:<30} "
          f"{row['total_return']:>8.1f}% {row['outperformance']:>9.1f}% {row['max_drawdown']:>9.1f}%")

  # Best strategies by timeframe across all coins
  print(f"\nBEST STRATEGY PER TIMEFRAME (ACROSS ALL COINS):")
  print("-" * 100)
  timeframe_best = results_df.groupby('timeframe').apply(
    lambda x: x.loc[x['total_return'].idxmax()]
  ).reset_index(drop=True)

  for _, row in timeframe_best.iterrows():
    print(f"{row['timeframe']:<10} | {row['coin']:<12} | {row['combination_name']:<30} | "
          f"{row['total_return']:>7.1f}% | Outperform: {row['outperformance']:>6.1f}%")

  # Strategy effectiveness by indicator count
  print(f"\nSTRATEGY COMPLEXITY ANALYSIS:")
  print("-" * 50)
  complexity_analysis = results_df.groupby('indicator_count').agg({
    'total_return': ['mean', 'max', 'std'],
    'outperformance': 'mean',
    'max_drawdown': 'mean'
  }).round(2)

  for count in sorted(results_df['indicator_count'].unique()):
    count_data = results_df[results_df['indicator_count'] == count]
    profitable_pct = len(count_data[count_data['total_return'] > 0]) / len(count_data) * 100

    print(f"{count} indicator(s): Avg return {count_data['total_return'].mean():.1f}%, "
          f"Max {count_data['total_return'].max():.1f}%, "
          f"Profitable: {profitable_pct:.1f}%")

  # WARNING SECTION
  print(f"\n{'=' * 100}")
  print("⚠️  CRITICAL WARNINGS & LIMITATIONS")
  print(f"{'=' * 100}")
  print("1. BACKTESTING BIAS: These results use perfect hindsight and may not predict future performance")
  print("2. OVERFITTING RISK: Testing many strategies increases chance of finding false patterns")
  print("3. TRANSACTION COSTS: Real trading includes fees, slippage, and spreads not modeled here")
  print("4. MARKET REGIME CHANGES: Past patterns may not continue in different market conditions")
  print("5. EXECUTION DIFFERENCES: Real trading involves delays, partial fills, and human psychology")
  print("6. SAMPLE SIZE: Results may not be statistically significant for some coin/strategy combinations")

  return results_df, coin_summaries_df


def create_coin_comparison_plots(results_df, coin_summaries_df):
  """Create comprehensive visualization comparing coins and strategies"""

  if len(results_df) == 0:
    return

  fig = plt.figure(figsize=(20, 16))

  # Plot 1: Coin performance comparison
  ax1 = plt.subplot(3, 3, 1)
  coin_summaries_sorted = coin_summaries_df.sort_values('best_strategy_return', ascending=True)
  y_pos = np.arange(len(coin_summaries_sorted))
  ax1.barh(y_pos, coin_summaries_sorted['best_strategy_return'])
  ax1.set_yticks(y_pos)
  ax1.set_yticklabels(coin_summaries_sorted['coin'])
  ax1.set_xlabel('Best Strategy Return (%)')
  ax1.set_title('Best Strategy Return by Coin')

  # Plot 2: Strategy success rate by coin
  ax2 = plt.subplot(3, 3, 2)
  success_rates = coin_summaries_df['profitable_strategies'] / coin_summaries_df['total_strategies_tested'] * 100
  ax2.bar(coin_summaries_df['coin'], success_rates)
  ax2.set_ylabel('Profitable Strategies (%)')
  ax2.set_title('Strategy Success Rate by Coin')
  ax2.tick_params(axis='x', rotation=45)

  # Plot 3: Average outperformance vs buy & hold
  ax3 = plt.subplot(3, 3, 3)
  ax3.bar(coin_summaries_df['coin'], coin_summaries_df['avg_outperformance'])
  ax3.set_ylabel('Avg Outperformance (%)')
  ax3.set_title('Average Outperformance vs Buy & Hold')
  ax3.tick_params(axis='x', rotation=45)
  ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

  # Plot 4: Risk vs Return by coin (scatter)
  ax4 = plt.subplot(3, 3, 4)
  colors = plt.cm.tab10(np.linspace(0, 1, len(coin_summaries_df)))
  for i, (_, coin) in enumerate(coin_summaries_df.iterrows()):
    coin_data = results_df[results_df['coin'] == coin['coin']]
    ax4.scatter(coin_data['max_drawdown'], coin_data['total_return'],
                alpha=0.6, label=coin['coin'], color=colors[i], s=30)
  ax4.set_xlabel('Max Drawdown (%)')
  ax4.set_ylabel('Total Return (%)')
  ax4.set_title('Risk vs Return by Coin')
  ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  ax4.grid(True, alpha=0.3)

  # Plot 5: Strategy complexity effectiveness
  ax5 = plt.subplot(3, 3, 5)
  complexity_data = results_df.groupby('indicator_count')['total_return'].mean()
  ax5.bar(complexity_data.index, complexity_data.values)
  ax5.set_xlabel('Number of Indicators')
  ax5.set_ylabel('Average Return (%)')
  ax5.set_title('Strategy Complexity vs Performance')

  # Plot 6: Timeframe effectiveness by coin
  ax6 = plt.subplot(3, 3, 6)
  timeframe_coin_performance = results_df.groupby(['coin', 'timeframe'])['total_return'].mean().unstack()
  timeframe_coin_performance.plot(kind='bar', ax=ax6, width=0.8)
  ax6.set_ylabel('Average Return (%)')
  ax6.set_title('Timeframe Performance by Coin')
  ax6.tick_params(axis='x', rotation=45)
  ax6.legend(title='Timeframe', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Plot 7: Distribution of returns
  ax7 = plt.subplot(3, 3, 7)
  ax7.hist(results_df['total_return'], bins=50, alpha=0.7, edgecolor='black')
  ax7.axvline(x=0, color='red', linestyle='--', alpha=0.8)
  ax7.set_xlabel('Total Return (%)')
  ax7.set_ylabel('Frequency')
  ax7.set_title('Distribution of All Strategy Returns')

  # Plot 8: Top performers by coin
  ax8 = plt.subplot(3, 3, 8)
  top_per_coin = results_df.groupby('coin')['total_return'].max().sort_values(ascending=True)
  y_pos = np.arange(len(top_per_coin))
  bars = ax8.barh(y_pos, top_per_coin.values)
  ax8.set_yticks(y_pos)
  ax8.set_yticklabels(top_per_coin.index)
  ax8.set_xlabel('Best Strategy Return (%)')
  ax8.set_title('Best Single Strategy per Coin')

  # Color bars based on performance
  for i, bar in enumerate(bars):
    if top_per_coin.values[i] > 50:
      bar.set_color('green')
    elif top_per_coin.values[i] > 0:
      bar.set_color('orange')
    else:
      bar.set_color('red')

  # Plot 9: Consistency vs Performance
  ax9 = plt.subplot(3, 3, 9)
  ax9.scatter(coin_summaries_df['avg_return'],
              coin_summaries_df['profitable_strategies'] / coin_summaries_df['total_strategies_tested'] * 100)
  ax9.set_xlabel('Average Return (%)')
  ax9.set_ylabel('Strategy Success Rate (%)')
  ax9.set_title('Consistency vs Performance')

  for _, coin in coin_summaries_df.iterrows():
    success_rate = coin['profitable_strategies'] / coin['total_strategies_tested'] * 100
    ax9.annotate(coin['coin'], (coin['avg_return'], success_rate), fontsize=8)

  plt.tight_layout()
  plt.show()


def save_comprehensive_results(results_df, coin_summaries_df, filename_prefix="multi_coin_simulation"):
  """Save comprehensive results to multiple CSV files"""
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  # Save detailed results
  results_filename = f"{filename_prefix}_detailed_{timestamp}.csv"
  results_df.to_csv(results_filename, index=False)

  # Save coin summaries
  summaries_filename = f"{filename_prefix}_coin_summaries_{timestamp}.csv"
  coin_summaries_df.to_csv(summaries_filename, index=False)

  # Save top strategies per coin
  top_strategies = results_df.groupby('coin').apply(
    lambda x: x.nlargest(5, 'total_return')
  ).reset_index(drop=True)
  top_filename = f"{filename_prefix}_top_strategies_{timestamp}.csv"
  top_strategies.to_csv(top_filename, index=False)

  print(f"\nResults saved:")
  print(f"  Detailed results: {results_filename}")
  print(f"  Coin summaries: {summaries_filename}")
  print(f"  Top strategies: {top_filename}")

  return results_filename, summaries_filename, top_filename


# Main execution
if __name__ == "__main__":
  # IMPORTANT: Update this path to your multi-coin CSV file
  file_path = "datasets/full_market_partial.csv"

  print("COMPREHENSIVE MULTI-COIN TRADING STRATEGY SIMULATION")
  print("=" * 60)
  print("⚠️  CRITICAL WARNINGS:")
  print("- This is for educational/research purposes ONLY")
  print("- Past performance does NOT guarantee future results")
  print("- Testing multiple coins MASSIVELY increases overfitting risk")
  print("- Real trading involves fees, slippage, and psychological factors")
  print("- Many 'profitable' strategies may be statistical flukes")
  print("- NEVER risk money you cannot afford to lose")
  print("=" * 60)

  # Optional: Test only specific coins (comment out to test all)
  # coins_to_test = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Major coins only
  coins_to_test = None  # Test all available coins

  try:
    # Run comprehensive simulation across all coins
    results_df, coin_summaries_df = run_comprehensive_simulation(
      file_path,
      initial_capital=1000,
      coins_to_test=coins_to_test
    )

    if not results_df.empty:
      print(f"\n{'=' * 100}")
      print("SIMULATION COMPLETED - ANALYZING RESULTS")
      print(f"{'=' * 100}")

      # Comprehensive analysis with coin profiling
      analyzed_results, coin_summaries = analyze_simulation_results(results_df, coin_summaries_df)

      # Save comprehensive results
      save_comprehensive_results(results_df, coin_summaries_df)

      # Create comprehensive visualizations
      create_coin_comparison_plots(results_df, coin_summaries_df)

      # Final critical reminder
      print(f"\n{'=' * 100}")
      print("⚠️  FINAL REMINDER: PROCEED WITH EXTREME CAUTION")
      print(f"{'=' * 100}")
      print("✓ Paper trade extensively before risking real money")
      print("✓ Start with tiny position sizes (0.1-0.5% of capital)")
      print("✓ Validate top strategies on NEW, unseen data")
      print("✓ Remember: Market conditions change constantly")
      print("✓ Most backtested strategies fail in live trading")
      print("✓ Consider this research, not investment advice")

      # Show data snooping bias warning
      total_tests = len(results_df)
      expected_false_positives = total_tests * 0.05  # 5% Type I error rate

      print(f"\nSTATISTICAL REALITY CHECK:")
      print(f"- Total strategies tested: {total_tests:,}")
      print(f"- Expected false positives (random chance): ~{expected_false_positives:.0f}")
      print(f"- Probability of finding 'amazing' strategy by chance: HIGH")
      print(f"- Recommendation: Be EXTREMELY skeptical of top performers")

    else:
      print("❌ No valid results generated. Check your data file and format.")
      print("Expected CSV format: Date, Symbol, Open, High, Low, Close, Volume")

  except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    print("Please update the file_path variable with your multi-coin CSV file")
    print("Required columns: Date, Symbol, Open, High, Low, Close, Volume")
    print("\nSample format:")
    print("Date,Symbol,Open,High,Low,Close,Volume")
    print("2022-10-20 00:00:00,BTCUSDT,19123.35,19165.34,18900.0,18972.69,11705.659")
    print("2022-10-20 00:00:00,ETHUSDT,1234.56,1245.67,1220.12,1235.89,25467.123")

  except Exception as e:
    print(f"❌ Error in simulation: {e}")
    print("\nPossible issues:")
    print("- Incorrect CSV format or column names")
    print("- Insufficient data for some coins")
    print("- Memory issues (too many coins/combinations)")
    print("- Date format problems")

    import traceback

    print(f"\nDetailed error:")
    traceback.print_exc()



