import pandas as pd
import numpy as np
import talib
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class IndicatorTester:
  def __init__(self, data_path, timeframe):
    """
    Initialize the indicator tester

    Args:
        data_path (str): Path to your CSV file
        timeframe (str): Timeframe of the data (5m, 15m, 30m, 1h)
    """
    self.data_path = data_path
    self.timeframe = timeframe
    self.df = None
    self.results = {}
    self.combination_results = {}

    # Define forward-looking periods based on timeframe
    self.forward_periods = {
      '5m': [1, 3, 5, 10],  # 5, 15, 25, 50 minutes
      '15m': [1, 2, 4, 8],  # 15, 30, 60, 120 minutes
      '30m': [1, 2, 4, 6],  # 30, 60, 120, 180 minutes
      '1h': [1, 3, 6, 12],  # 1, 3, 6, 12 hours
      '2h': [1, 2, 3, 6],  # 2, 4, 6, 12 hours
      '4h': [1, 2, 3, 6],  # 4, 8, 12, 24 hours
      '6h': [1, 2, 4, 8],  # 6, 12, 24, 48 hours
      '8h': [1, 2, 3, 6],  # 8, 16, 24, 48 hours
      '12h': [1, 2, 4, 7],  # 12, 24, 48, 84 hours
      '1d': [1, 2, 3, 7]  # 1, 2, 3, 7 days
    }

    print(f"=" * 60)
    print(f"INITIALIZING INDICATOR TESTER FOR {timeframe.upper()} TIMEFRAME")
    print(f"=" * 60)

  def load_data(self):
    """Load and prepare the data"""
    print(f"\n📊 Loading data from: {self.data_path}")

    # Assuming standard OHLCV format - adjust column names as needed
    self.df = pd.read_csv(self.data_path)

    # Standardize column names (adjust these based on your actual column names)
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if len(self.df.columns) >= 6:
      self.df.columns = expected_columns[:len(self.df.columns)]

    # Convert timestamp if needed
    if 'timestamp' in self.df.columns:
      self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
      self.df.set_index('timestamp', inplace=True)

    # Limit to last 30,000 rows for performance
    if len(self.df) > 30000:
      print(f"   🔄 Limiting data to last 30,000 rows (from {len(self.df)} total rows)")
      self.df = self.df.tail(30000).copy()

    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {self.df.shape}")
    print(f"   Columns: {list(self.df.columns)}")
    print(f"   Date range: {self.df.index[0]} to {self.df.index[-1]}")
    print(f"   Missing values: {self.df.isnull().sum().sum()}")

    return self.df

  def calculate_indicators(self):
    """Calculate comprehensive set of indicators including your previous momentum studies"""
    print(f"\n🔧 Calculating comprehensive technical indicators (including your momentum studies)...")

    high = self.df['high'].values
    low = self.df['low'].values
    close = self.df['close'].values
    volume = self.df['volume'].values

    # Convert to pandas for easier calculation of some indicators
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    volume_series = pd.Series(volume)

    indicators = {}

    try:
      # === YOUR PREVIOUS MOMENTUM INDICATORS ===
      print("   📈 Calculating momentum indicators (your previous promising ones)...")

      # Multiple Moving Averages (from your script)
      indicators['SMA_5'] = talib.SMA(close, timeperiod=5)
      indicators['SMA_10'] = talib.SMA(close, timeperiod=10)
      indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
      indicators['SMA_50'] = talib.SMA(close, timeperiod=50)

      # Multiple Exponential Moving Averages (from your script)
      indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
      indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
      indicators['EMA_50'] = talib.EMA(close, timeperiod=50)

      # Price Momentum - Multiple Periods (YOUR KEY INDICATORS!)
      indicators['Momentum_5'] = (close_series / close_series.shift(5) - 1).values
      indicators['Momentum_10'] = (close_series / close_series.shift(10) - 1).values
      indicators['Momentum_20'] = (close_series / close_series.shift(20) - 1).values

      # Rate of Change - Multiple Periods (from your script)
      indicators['ROC_5'] = ((close_series - close_series.shift(5)) / close_series.shift(5) * 100).values
      indicators['ROC_10'] = ((close_series - close_series.shift(10)) / close_series.shift(10) * 100).values

      # Volume indicators (from your script)
      indicators['Volume_SMA'] = volume_series.rolling(window=20).mean().values
      indicators['Volume_ratio'] = (volume_series / volume_series.rolling(window=20).mean()).values

      # === ENHANCED BOLLINGER BANDS ===
      bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
      indicators['BB_Upper'] = bb_upper
      indicators['BB_Lower'] = bb_lower
      indicators['BB_Middle'] = bb_middle
      indicators['BB_width'] = bb_upper - bb_lower
      indicators['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # Your BB position indicator

      # === MACD WITH HISTOGRAM ===
      macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
      indicators['MACD'] = macd
      indicators['MACD_Signal'] = macdsignal
      indicators['MACD_histogram'] = macdhist  # You had this in your script

      # === CROSSOVER SIGNALS (from your script) ===
      indicators['SMA_5_20_cross'] = np.where(indicators['SMA_5'] > indicators['SMA_20'], 1, 0)
      indicators['EMA_12_26_cross'] = np.where(indicators['EMA_12'] > indicators['EMA_26'], 1, 0)
      indicators['MACD_signal_cross'] = np.where(macd > macdsignal, 1, 0)

      # === STANDARD OSCILLATORS ===
      indicators['RSI'] = talib.RSI(close, timeperiod=14)

      # Stochastic (matching your calculation)
      slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
      indicators['STOCH_K'] = slowk
      indicators['STOCH_D'] = slowd

      # Williams %R (matching your calculation method)
      indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

      # CCI (matching your calculation)
      indicators['CCI'] = talib.CCI(high, low, close, timeperiod=20)  # Using 20 period like your script

      # === ADDITIONAL POWERFUL INDICATORS ===
      indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
      indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
      indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
      indicators['OBV'] = talib.OBV(close, volume)
      indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

      # VWAP
      indicators['VWAP'] = (close * volume).cumsum() / volume.cumsum()

      # Keltner Channels
      ema_20 = talib.EMA(close, timeperiod=20)
      atr_10 = talib.ATR(high, low, close, timeperiod=10)
      indicators['Keltner_Upper'] = ema_20 + (2 * atr_10)
      indicators['Keltner_Lower'] = ema_20 - (2 * atr_10)

      # Aroon
      aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
      indicators['Aroon_Up'] = aroonup
      indicators['Aroon_Down'] = aroondown

      # Advanced oscillators
      indicators['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
      indicators['TRIX'] = talib.TRIX(close, timeperiod=14)

      print(f"✅ All indicators calculated successfully!")
      print(f"   📊 Total indicators: {len(indicators)}")
      print(f"   🚀 Including your momentum studies: Momentum_5, Momentum_10, Momentum_20")
      print(f"   📈 Including your crossover signals: SMA_5_20, EMA_12_26, MACD_signal")

    except Exception as e:
      print(f"❌ Error calculating indicators: {e}")

    return indicators

  def generate_signals(self, indicators):
    """Generate buy/sell signals for each indicator"""
    print(f"\n🎯 Generating trading signals (including your momentum strategies)...")

    signals = {}
    close = self.df['close'].values

    # === YOUR PREVIOUS MOMENTUM SIGNALS (the promising ones!) ===
    print("   🚀 Generating momentum signals...")

    # Multiple SMA Signals
    signals['SMA_5'] = np.where(close > indicators['SMA_5'], 1, -1)
    signals['SMA_10'] = np.where(close > indicators['SMA_10'], 1, -1)
    signals['SMA_20'] = np.where(close > indicators['SMA_20'], 1, -1)
    signals['SMA_50'] = np.where(close > indicators['SMA_50'], 1, -1)

    # Multiple EMA Signals
    signals['EMA_12'] = np.where(close > indicators['EMA_12'], 1, -1)
    signals['EMA_26'] = np.where(close > indicators['EMA_26'], 1, -1)
    signals['EMA_50'] = np.where(close > indicators['EMA_50'], 1, -1)

    # YOUR KEY MOMENTUM SIGNALS - Different thresholds for strong/weak momentum
    # Strong momentum signals (your promising approach!)
    signals['Momentum_5_Strong'] = np.where(indicators['Momentum_5'] > 0.02, 1,  # Strong positive momentum
                                            np.where(indicators['Momentum_5'] < -0.02, -1,
                                                     0))  # Strong negative momentum
    signals['Momentum_10_Strong'] = np.where(indicators['Momentum_10'] > 0.03, 1,
                                             np.where(indicators['Momentum_10'] < -0.03, -1, 0))
    signals['Momentum_20_Strong'] = np.where(indicators['Momentum_20'] > 0.05, 1,
                                             np.where(indicators['Momentum_20'] < -0.05, -1, 0))

    # Weak momentum signals (for comparison)
    signals['Momentum_5_Weak'] = np.where(indicators['Momentum_5'] > 0, 1, -1)
    signals['Momentum_10_Weak'] = np.where(indicators['Momentum_10'] > 0, 1, -1)
    signals['Momentum_20_Weak'] = np.where(indicators['Momentum_20'] > 0, 1, -1)

    # ROC Signals - Multiple periods
    signals['ROC_5'] = np.where(indicators['ROC_5'] > 0, 1, -1)
    signals['ROC_10'] = np.where(indicators['ROC_10'] > 0, 1, -1)

    # Volume signals
    signals['Volume_High'] = np.where(indicators['Volume_ratio'] > 1.5, 1,  # High volume confirmation
                                      np.where(indicators['Volume_ratio'] < 0.5, -1, 0))  # Low volume warning

    # YOUR CROSSOVER SIGNALS (from your script)
    signals['SMA_5_20_Cross'] = indicators['SMA_5_20_cross'] * 2 - 1  # Convert 0,1 to -1,1
    signals['EMA_12_26_Cross'] = indicators['EMA_12_26_cross'] * 2 - 1
    signals['MACD_Signal_Cross'] = indicators['MACD_signal_cross'] * 2 - 1

    # Enhanced Bollinger Bands (with your BB_position)
    signals['BB_Standard'] = np.where(close < indicators['BB_Lower'], 1,
                                      np.where(close > indicators['BB_Upper'], -1, 0))
    signals['BB_Position'] = np.where(indicators['BB_position'] < 0.2, 1,  # Near lower band
                                      np.where(indicators['BB_position'] > 0.8, -1, 0))  # Near upper band
    signals['BB_Width'] = np.where(indicators['BB_width'] < np.nanpercentile(indicators['BB_width'], 20), 1,
                                   -1)  # Low volatility

    # MACD with Histogram
    signals['MACD'] = np.where(indicators['MACD'] > indicators['MACD_Signal'], 1, -1)
    signals['MACD_Histogram'] = np.where(indicators['MACD_histogram'] > 0, 1, -1)

    # === STANDARD OSCILLATOR SIGNALS ===
    signals['RSI'] = np.where(indicators['RSI'] < 30, 1, np.where(indicators['RSI'] > 70, -1, 0))
    signals['STOCH'] = np.where(indicators['STOCH_K'] < 20, 1, np.where(indicators['STOCH_K'] > 80, -1, 0))
    signals['CCI'] = np.where(indicators['CCI'] < -100, 1, np.where(indicators['CCI'] > 100, -1, 0))
    signals['WILLR'] = np.where(indicators['WILLR'] < -80, 1, np.where(indicators['WILLR'] > -20, -1, 0))
    signals['MFI'] = np.where(indicators['MFI'] < 20, 1, np.where(indicators['MFI'] > 80, -1, 0))
    signals['ULTOSC'] = np.where(indicators['ULTOSC'] < 30, 1, np.where(indicators['ULTOSC'] > 70, -1, 0))

    # === TREND AND VOLATILITY SIGNALS ===
    signals['ADX'] = np.where(indicators['ADX'] > 25, 1, -1)
    signals['SAR'] = np.where(close > indicators['SAR'], 1, -1)
    signals['VWAP'] = np.where(close > indicators['VWAP'], 1, -1)

    # Volume momentum
    obv_ma = pd.Series(indicators['OBV']).rolling(10).mean().values
    signals['OBV'] = np.where(indicators['OBV'] > obv_ma, 1, -1)

    # Keltner Channels
    signals['Keltner'] = np.where(close < indicators['Keltner_Lower'], 1,
                                  np.where(close > indicators['Keltner_Upper'], -1, 0))

    # Aroon
    signals['Aroon'] = np.where(indicators['Aroon_Up'] > indicators['Aroon_Down'], 1, -1)

    # TRIX
    trix_signal = pd.Series(indicators['TRIX']).rolling(9).mean().values
    signals['TRIX'] = np.where(indicators['TRIX'] > trix_signal, 1, -1)

    # ATR volatility
    atr_ma = pd.Series(indicators['ATR']).rolling(10).mean().values
    signals['ATR'] = np.where(indicators['ATR'] < atr_ma, 1, -1)

    print(f"✅ Signals generated for {len(signals)} indicators")
    print(f"   🎯 Including your momentum strategies: Strong vs Weak momentum thresholds")
    print(f"   📊 Including your crossover signals: SMA_5_20, EMA_12_26, MACD_Signal")

    return signals

  def calculate_forward_returns(self):
    """Calculate forward returns for different periods"""
    print(f"\n📈 Calculating forward returns...")

    close = self.df['close'].values
    forward_returns = {}

    periods = self.forward_periods[self.timeframe]

    for period in periods:
      if period < len(close):
        returns = np.full(len(close), np.nan)
        returns[:-period] = (close[period:] / close[:-period]) - 1
        forward_returns[f'return_{period}p'] = returns

    print(f"✅ Forward returns calculated for periods: {periods}")
    return forward_returns

  def detect_market_regime(self):
    """Detect market regimes (trending vs ranging)"""
    print(f"\n🌊 Detecting market regimes...")

    close = self.df['close'].values

    # Calculate ADX for trend strength
    high = self.df['high'].values
    low = self.df['low'].values
    adx = talib.ADX(high, low, close, timeperiod=14)

    # Calculate volatility (ATR)
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_normalized = atr / close

    # Define regimes
    trending = adx > 25
    high_vol = atr_normalized > np.nanpercentile(atr_normalized, 75)

    regimes = np.where(trending & ~high_vol, 'trending_stable',
                       np.where(trending & high_vol, 'trending_volatile',
                                np.where(~trending & high_vol, 'ranging_volatile', 'ranging_stable')))

    regime_counts = pd.Series(regimes).value_counts()
    print(f"✅ Market regimes detected:")
    for regime, count in regime_counts.items():
      print(f"   {regime}: {count} periods ({count / len(regimes) * 100:.1f}%)")

    return regimes

  def test_individual_indicators(self, signals, forward_returns, regimes):
    """Test each indicator individually"""
    print(f"\n🔍 Testing individual indicator performance...")
    print(f"=" * 80)

    periods = self.forward_periods[self.timeframe]
    results = {}

    for indicator_name, signal_array in signals.items():
      print(f"\n📊 Testing {indicator_name}:")
      print(f"-" * 50)

      indicator_results = {
        'total_signals': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'accuracy_by_period': {},
        'accuracy_by_regime': {},
        'performance_metrics': {}
      }

      # Count signals
      buy_signals = signal_array == 1
      sell_signals = signal_array == -1
      total_signals = np.sum(buy_signals | sell_signals)

      indicator_results['total_signals'] = total_signals
      indicator_results['buy_signals'] = np.sum(buy_signals)
      indicator_results['sell_signals'] = np.sum(sell_signals)

      print(f"   Total signals: {total_signals}")
      print(f"   Buy signals: {np.sum(buy_signals)} ({np.sum(buy_signals) / total_signals * 100:.1f}%)")
      print(f"   Sell signals: {np.sum(sell_signals)} ({np.sum(sell_signals) / total_signals * 100:.1f}%)")

      if total_signals < 10:
        print(f"   ⚠️  Warning: Too few signals for reliable analysis")
        results[indicator_name] = indicator_results
        continue

      # Test accuracy for different forward periods
      print(f"\n   📈 Accuracy by forward period:")
      for period in periods:
        if f'return_{period}p' in forward_returns:
          returns = forward_returns[f'return_{period}p']

          # Calculate directional accuracy
          buy_mask = buy_signals & ~np.isnan(returns)
          sell_mask = sell_signals & ~np.isnan(returns)

          if np.sum(buy_mask) > 0:
            buy_accuracy = np.mean(returns[buy_mask] > 0) * 100
            avg_buy_return = np.mean(returns[buy_mask]) * 100
          else:
            buy_accuracy = avg_buy_return = 0

          if np.sum(sell_mask) > 0:
            sell_accuracy = np.mean(returns[sell_mask] < 0) * 100
            avg_sell_return = np.mean(returns[sell_mask]) * 100
          else:
            sell_accuracy = avg_sell_return = 0

          overall_accuracy = (buy_accuracy * np.sum(buy_mask) + sell_accuracy * np.sum(sell_mask)) / (
              np.sum(buy_mask) + np.sum(sell_mask)) if (np.sum(buy_mask) + np.sum(sell_mask)) > 0 else 0

          indicator_results['accuracy_by_period'][period] = {
            'overall_accuracy': overall_accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'avg_buy_return': avg_buy_return,
            'avg_sell_return': avg_sell_return
          }

          print(
            f"      {period} periods: {overall_accuracy:.1f}% accuracy, Avg return: {(avg_buy_return + avg_sell_return) / 2:.2f}%")

      # Test accuracy by market regime
      print(f"\n   🌊 Accuracy by market regime:")
      for regime in np.unique(regimes):
        regime_mask = regimes == regime
        regime_buy_mask = buy_signals & regime_mask & ~np.isnan(forward_returns[f'return_{periods[0]}p'])
        regime_sell_mask = sell_signals & regime_mask & ~np.isnan(forward_returns[f'return_{periods[0]}p'])

        if np.sum(regime_buy_mask | regime_sell_mask) > 5:  # Minimum signals for regime analysis
          returns = forward_returns[f'return_{periods[0]}p']

          if np.sum(regime_buy_mask) > 0:
            regime_buy_acc = np.mean(returns[regime_buy_mask] > 0) * 100
          else:
            regime_buy_acc = 0

          if np.sum(regime_sell_mask) > 0:
            regime_sell_acc = np.mean(returns[regime_sell_mask] < 0) * 100
          else:
            regime_sell_acc = 0

          regime_overall = (regime_buy_acc * np.sum(regime_buy_mask) + regime_sell_acc * np.sum(regime_sell_mask)) / (
              np.sum(regime_buy_mask) + np.sum(regime_sell_mask)) if (np.sum(regime_buy_mask) + np.sum(
            regime_sell_mask)) > 0 else 0

          indicator_results['accuracy_by_regime'][regime] = {
            'accuracy': regime_overall,
            'signals': np.sum(regime_buy_mask | regime_sell_mask)
          }

          print(
            f"      {regime}: {regime_overall:.1f}% accuracy ({np.sum(regime_buy_mask | regime_sell_mask)} signals)")

      results[indicator_name] = indicator_results

    return results

  def test_indicator_combinations(self, signals, forward_returns, max_combinations=50):
    """Test combinations of 2-3 indicators"""
    print(f"\n🔗 Testing indicator combinations...")
    print(f"=" * 80)

    indicator_names = list(signals.keys())
    combination_results = {}

    periods = self.forward_periods[self.timeframe]
    primary_period = periods[0]  # Use first period for combination testing
    returns = forward_returns[f'return_{primary_period}p']

    # Test 2-indicator combinations
    print(f"\n🔄 Testing 2-indicator combinations (showing top {max_combinations // 2}):")
    two_combos = list(combinations(indicator_names, 2))

    two_results = []

    for combo in two_combos[:max_combinations]:  # Limit to prevent excessive computation
      ind1, ind2 = combo

      # Unanimous agreement strategy
      unanimous_buy = (signals[ind1] == 1) & (signals[ind2] == 1)
      unanimous_sell = (signals[ind1] == -1) & (signals[ind2] == -1)

      buy_mask = unanimous_buy & ~np.isnan(returns)
      sell_mask = unanimous_sell & ~np.isnan(returns)

      total_signals = np.sum(buy_mask | sell_mask)

      if total_signals >= 5:  # Minimum signals for analysis
        if np.sum(buy_mask) > 0:
          buy_accuracy = np.mean(returns[buy_mask] > 0) * 100
          avg_buy_return = np.mean(returns[buy_mask]) * 100
        else:
          buy_accuracy = avg_buy_return = 0

        if np.sum(sell_mask) > 0:
          sell_accuracy = np.mean(returns[sell_mask] < 0) * 100
          avg_sell_return = np.mean(returns[sell_mask]) * 100
        else:
          sell_accuracy = avg_sell_return = 0

        overall_accuracy = (buy_accuracy * np.sum(buy_mask) + sell_accuracy * np.sum(
          sell_mask)) / total_signals if total_signals > 0 else 0
        avg_return = (avg_buy_return + avg_sell_return) / 2

        two_results.append({
          'combination': f"{ind1} + {ind2}",
          'accuracy': overall_accuracy,
          'total_signals': total_signals,
          'avg_return': avg_return,
          'buy_signals': np.sum(buy_mask),
          'sell_signals': np.sum(sell_mask)
        })

    # Sort and display top combinations
    two_results.sort(key=lambda x: x['accuracy'], reverse=True)

    for i, result in enumerate(two_results[:20]):  # Show top 20
      print(f"   {i + 1:2d}. {result['combination']:<30} | "
            f"Accuracy: {result['accuracy']:5.1f}% | "
            f"Signals: {result['total_signals']:3d} | "
            f"Avg Return: {result['avg_return']:+5.2f}%")

    combination_results['two_indicator'] = two_results[:20]

    # Test 3-indicator combinations (limited number)
    print(f"\n🔄 Testing 3-indicator combinations (showing top {max_combinations // 4}):")
    three_combos = list(combinations(indicator_names, 3))

    three_results = []

    for combo in three_combos[:max_combinations // 2]:  # Even more limited for 3-combinations
      ind1, ind2, ind3 = combo

      # Unanimous agreement strategy
      unanimous_buy = (signals[ind1] == 1) & (signals[ind2] == 1) & (signals[ind3] == 1)
      unanimous_sell = (signals[ind1] == -1) & (signals[ind2] == -1) & (signals[ind3] == -1)

      buy_mask = unanimous_buy & ~np.isnan(returns)
      sell_mask = unanimous_sell & ~np.isnan(returns)

      total_signals = np.sum(buy_mask | sell_mask)

      if total_signals >= 3:  # Minimum signals for analysis
        if np.sum(buy_mask) > 0:
          buy_accuracy = np.mean(returns[buy_mask] > 0) * 100
          avg_buy_return = np.mean(returns[buy_mask]) * 100
        else:
          buy_accuracy = avg_buy_return = 0

        if np.sum(sell_mask) > 0:
          sell_accuracy = np.mean(returns[sell_mask] < 0) * 100
          avg_sell_return = np.mean(returns[sell_mask]) * 100
        else:
          sell_accuracy = avg_sell_return = 0

        overall_accuracy = (buy_accuracy * np.sum(buy_mask) + sell_accuracy * np.sum(
          sell_mask)) / total_signals if total_signals > 0 else 0
        avg_return = (avg_buy_return + avg_sell_return) / 2

        three_results.append({
          'combination': f"{ind1} + {ind2} + {ind3}",
          'accuracy': overall_accuracy,
          'total_signals': total_signals,
          'avg_return': avg_return,
          'buy_signals': np.sum(buy_mask),
          'sell_signals': np.sum(sell_mask)
        })

    # Sort and display top combinations
    three_results.sort(key=lambda x: x['accuracy'], reverse=True)

    for i, result in enumerate(three_results[:15]):  # Show top 15
      print(f"   {i + 1:2d}. {result['combination']:<45} | "
            f"Accuracy: {result['accuracy']:5.1f}% | "
            f"Signals: {result['total_signals']:3d} | "
            f"Avg Return: {result['avg_return']:+5.2f}%")

    combination_results['three_indicator'] = three_results[:15]

    return combination_results

  def generate_summary_report(self, individual_results, combination_results):
    """Generate a comprehensive summary report"""
    print(f"\n" + "=" * 80)
    print(f"📋 COMPREHENSIVE SUMMARY REPORT - {self.timeframe.upper()} TIMEFRAME")
    print(f"=" * 80)

    # Top performing individual indicators
    print(f"\n🏆 TOP 15 INDIVIDUAL INDICATORS:")
    print(f"-" * 70)

    # Sort indicators by best overall accuracy (using first period)
    periods = self.forward_periods[self.timeframe]
    primary_period = periods[0]

    individual_rankings = []
    for ind_name, results in individual_results.items():
      if primary_period in results['accuracy_by_period']:
        acc = results['accuracy_by_period'][primary_period]['overall_accuracy']
        signals = results['total_signals']
        avg_return = (results['accuracy_by_period'][primary_period]['avg_buy_return'] +
                      results['accuracy_by_period'][primary_period]['avg_sell_return']) / 2

        individual_rankings.append({
          'indicator': ind_name,
          'accuracy': acc,
          'signals': signals,
          'avg_return': avg_return
        })

    individual_rankings.sort(key=lambda x: x['accuracy'], reverse=True)

    for i, result in enumerate(individual_rankings[:15]):
      # Highlight momentum indicators
      indicator_name = result['indicator']
      if 'Momentum' in indicator_name:
        indicator_name = f"🚀 {indicator_name}"
      elif 'Cross' in indicator_name:
        indicator_name = f"📈 {indicator_name}"

      print(f"   {i + 1:2d}. {indicator_name:<25} | "
            f"Accuracy: {result['accuracy']:5.1f}% | "
            f"Signals: {result['signals']:4d} | "
            f"Avg Return: {result['avg_return']:+5.2f}%")

    # === MOMENTUM STRATEGY ANALYSIS ===
    print(f"\n🚀 MOMENTUM STRATEGY ANALYSIS (Your Previous Focus):")
    print(f"-" * 70)

    momentum_indicators = [r for r in individual_rankings if 'Momentum' in r['indicator']]
    if momentum_indicators:
      print(f"   📊 Momentum Indicator Performance:")
      for i, result in enumerate(momentum_indicators):
        strength = "Strong" if "Strong" in result['indicator'] else "Weak"
        period = result['indicator'].split('_')[1]
        print(f"      {strength} {period}p momentum: {result['accuracy']:5.1f}% accuracy, "
              f"{result['signals']:3d} signals, {result['avg_return']:+5.2f}% avg return")

    # Compare strong vs weak momentum
    strong_momentum = [r for r in momentum_indicators if 'Strong' in r['indicator']]
    weak_momentum = [r for r in momentum_indicators if 'Weak' in r['indicator']]

    if strong_momentum and weak_momentum:
      avg_strong_acc = np.mean([r['accuracy'] for r in strong_momentum])
      avg_weak_acc = np.mean([r['accuracy'] for r in weak_momentum])
      print(f"\n   🎯 Strong vs Weak Momentum Comparison:")
      print(f"      Strong momentum average: {avg_strong_acc:.1f}% accuracy")
      print(f"      Weak momentum average: {avg_weak_acc:.1f}% accuracy")
      print(f"      Difference: {avg_strong_acc - avg_weak_acc:+.1f}% (Strong momentum advantage)")

    # === CROSSOVER STRATEGY ANALYSIS ===
    crossover_indicators = [r for r in individual_rankings if 'Cross' in r['indicator']]
    if crossover_indicators:
      print(f"\n📈 CROSSOVER STRATEGY ANALYSIS:")
      print(f"-" * 50)
      for result in crossover_indicators:
        print(f"      {result['indicator']}: {result['accuracy']:5.1f}% accuracy, "
              f"{result['signals']:3d} signals")

    # Best 2-indicator combinations
    print(f"\n🤝 TOP 8 TWO-INDICATOR COMBINATIONS:")
    print(f"-" * 70)

    if 'two_indicator' in combination_results:
      for i, result in enumerate(combination_results['two_indicator'][:8]):
        combo_name = result['combination']
        # Highlight momentum combinations
        if 'Momentum' in combo_name:
          combo_name = f"🚀 {combo_name}"
        elif any(x in combo_name for x in ['Cross', 'SMA_', 'EMA_']):
          combo_name = f"📈 {combo_name}"

        print(f"   {i + 1}. {combo_name:<45} | "
              f"Accuracy: {result['accuracy']:5.1f}% | "
              f"Signals: {result['total_signals']:3d}")

    # Best 3-indicator combinations
    print(f"\n🎯 TOP 5 THREE-INDICATOR COMBINATIONS:")
    print(f"-" * 70)

    if 'three_indicator' in combination_results:
      for i, result in enumerate(combination_results['three_indicator'][:5]):
        combo_name = result['combination']
        if 'Momentum' in combo_name:
          combo_name = f"🚀 {combo_name}"

        print(f"   {i + 1}. {combo_name:<55} | "
              f"Accuracy: {result['accuracy']:5.1f}% | "
              f"Signals: {result['total_signals']:3d}")

    # Data quality and reliability notes
    print(f"\n📊 DATA QUALITY & RELIABILITY NOTES:")
    print(f"-" * 60)
    total_periods = len(self.df)
    print(f"   • Total data points: {total_periods}")
    print(f"   • Timeframe: {self.timeframe}")
    print(f"   • Analysis periods: {self.forward_periods[self.timeframe]}")
    print(f"   • Minimum signals for reliable analysis: 10+ (individual), 5+ (combinations)")
    print(f"   • Strong momentum threshold: 2-5% depending on period")
    print(f"   • Weak momentum threshold: Any positive/negative direction")

    # Regime-specific insights
    print(f"\n🌊 KEY INSIGHTS BY MARKET REGIME:")
    print(f"-" * 60)

    regime_performance = {}
    for ind_name, results in individual_results.items():
      for regime, regime_data in results.get('accuracy_by_regime', {}).items():
        if regime not in regime_performance:
          regime_performance[regime] = []
        regime_performance[regime].append({
          'indicator': ind_name,
          'accuracy': regime_data['accuracy'],
          'signals': regime_data['signals']
        })

    for regime, performances in regime_performance.items():
      if performances:
        performances.sort(key=lambda x: x['accuracy'], reverse=True)
        best = performances[0]
        best_momentum = next((p for p in performances if 'Momentum' in p['indicator']), None)

        print(f"   • {regime.replace('_', ' ').title()}:")
        print(f"     Best overall: {best['indicator']} ({best['accuracy']:.1f}% accuracy)")
        if best_momentum and best_momentum != best:
          print(f"     Best momentum: {best_momentum['indicator']} ({best_momentum['accuracy']:.1f}% accuracy)")

    print(f"\n" + "=" * 80)
    print(f"✅ ANALYSIS COMPLETE FOR {self.timeframe.upper()} TIMEFRAME")
    print(f"🚀 Pay special attention to momentum indicators - they were your previous winners!")
    print(f"=" * 80)

  def run_complete_analysis(self):
    """Run the complete analysis pipeline"""
    print(f"\n🚀 STARTING COMPLETE INDICATOR ANALYSIS")
    print(f"Timeframe: {self.timeframe}")
    print(f"Data source: {self.data_path}")

    # Load and prepare data
    self.load_data()

    # Calculate all indicators
    indicators = self.calculate_indicators()

    # Generate signals
    signals = self.generate_signals(indicators)

    # Calculate forward returns
    forward_returns = self.calculate_forward_returns()

    # Detect market regimes
    regimes = self.detect_market_regime()

    # Test individual indicators
    individual_results = self.test_individual_indicators(signals, forward_returns, regimes)

    # Test indicator combinations
    combination_results = self.test_indicator_combinations(signals, forward_returns)

    # Generate summary report
    self.generate_summary_report(individual_results, combination_results)

    # Store results for later analysis
    self.results = individual_results
    self.combination_results = combination_results

    return {
      'individual_results': individual_results,
      'combination_results': combination_results,
      'timeframe': self.timeframe,
      'data_points': len(self.df)
    }


# Usage example:
if __name__ == "__main__":
  # Example usage - replace with your actual file paths and timeframes

  # For 5-minute data
  # tester_5m = IndicatorTester("path/to/your/eth_5m_data.csv", "5m")
  # results_5m = tester_5m.run_complete_analysis()

  # For 15-minute data
  # tester_15m = IndicatorTester("path/to/your/eth_15m_data.csv", "15m")
  # results_15m = tester_15m.run_complete_analysis()

  # For 30-minute data
  # tester_30m = IndicatorTester("path/to/your/eth_30m_data.csv", "30m")
  # results_30m = tester_30m.run_complete_analysis()

  # For 1-hour data
  # tester_1h = IndicatorTester("path/to/your/eth_1h_data.csv", "1h")
  # results_1h = tester_1h.run_complete_analysis()

  # For 2-hour data
  # tester_2h = IndicatorTester("ETHUSDT_2h.csv", "2h")
  # results_2h = tester_2h.run_complete_analysis()

  # For 4-hour data
  # tester_4h = IndicatorTester("ETHUSDT_4h.csv", "4h")
  # results_4h = tester_4h.run_complete_analysis()

  # For 6-hour data
  # tester_6h = IndicatorTester("ETHUSDT_6h.csv", "6h")
  # results_6h = tester_6h.run_complete_analysis()

  # For 8-hour data
  # tester_8h = IndicatorTester("ETHUSDT_8h.csv", "8h")
  # results_8h = tester_8h.run_complete_analysis()

  # For 12-hour data
  # tester_12h = IndicatorTester("ETHUSDT_12h.csv", "12h")
  # results_12h = tester_12h.run_complete_analysis()

  # For daily data
  tester_1d = IndicatorTester("datasets/ETHUSDT_1d.csv", "1d")
  results_1d = tester_1d.run_complete_analysis()

  print("\nTo run the analysis, create an instance with your data:")
  print("tester = IndicatorTester('your_file.csv', '5m')  # or '15m', '30m', '1h'")
  print("results = tester.run_complete_analysis()")