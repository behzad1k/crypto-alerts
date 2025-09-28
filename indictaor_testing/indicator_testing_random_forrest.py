import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
  """Load 1h data and prepare for analysis"""
  df = pd.read_csv(file_path)

  # Standardize column names
  column_mapping = {
    'timestamp': 'Date',
    'Timestamp': 'Date',
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
  }

  for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
      df = df.rename(columns={old_name: new_name})

  # Handle timestamp conversion
  if 'Date' in df.columns:
    try:
      df['Date'] = pd.to_datetime(df['Date'])
    except:
      try:
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
      except:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')

  df = df.sort_values('Date').reset_index(drop=True)
  df.set_index('Date', inplace=True)

  return df


def create_timeframes(df_1h):
  """Create 4h, 6h, and 8h timeframes from 1h data"""
  timeframes = {}

  # 2h timeframe
  timeframes['2h'] = df_1h.resample('2H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
  }).dropna()

  # 4h timeframe
  timeframes['4h'] = df_1h.resample('4H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
  }).dropna()

  # 6h timeframe
  timeframes['6h'] = df_1h.resample('6H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
  }).dropna()

  # 8h timeframe
  timeframes['8h'] = df_1h.resample('8H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
  }).dropna()

  return timeframes


def calculate_indicators(df):
  """Calculate the 5 specified indicators"""
  data = df.copy()

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


def generate_signals(data):
  """Generate buy/sell signals for each indicator"""
  signals = pd.DataFrame(index=data.index)

  # MACD crossover signals
  macd_prev = data['MACD'].shift(1)
  macd_signal_prev = data['MACD_signal'].shift(1)

  bullish_cross = (data['MACD'] > data['MACD_signal']) & (macd_prev <= macd_signal_prev)
  bearish_cross = (data['MACD'] < data['MACD_signal']) & (macd_prev >= macd_signal_prev)

  signals['MACD'] = np.where(bullish_cross, 1, np.where(bearish_cross, -1, 0))

  # Momentum signals
  signals['Momentum_5'] = np.where(data['Momentum_5'] > 0.005, 1,
                                   np.where(data['Momentum_5'] < -0.005, -1, 0))

  signals['Momentum_10'] = np.where(data['Momentum_10'] > 0.008, 1,
                                    np.where(data['Momentum_10'] < -0.008, -1, 0))

  # Williams %R signals
  signals['WILLR'] = np.where(data['WILLR'] < -80, 1,
                              np.where(data['WILLR'] > -20, -1, 0))

  # Stochastic D signals
  signals['STOCH_D'] = np.where(data['STOCH_D'] < 20, 1,
                                np.where(data['STOCH_D'] > 80, -1, 0))

  return signals


def create_targets(data):
  """Create simple up/down targets for next period"""
  targets = pd.DataFrame(index=data.index)

  # Next period up/down
  future_return = (data['Close'].shift(-1) / data['Close'] - 1)
  targets['next_up'] = (future_return > 0).astype(int)

  return targets


def test_individual_indicators(signals, targets, timeframe_name):
  """Test individual indicator performance"""
  results = {}
  indicators = ['MACD', 'Momentum_5', 'Momentum_10', 'WILLR', 'STOCH_D']

  print(f"\n{timeframe_name} Individual Indicator Results:")
  print("-" * 50)

  for indicator in indicators:
    try:
      # Get buy and sell signals
      buy_signals = signals[indicator] == 1
      sell_signals = signals[indicator] == -1

      if buy_signals.sum() > 5 and sell_signals.sum() > 5:
        # Calculate buy accuracy
        buy_targets = targets['next_up'][buy_signals]
        buy_accuracy = buy_targets.mean()

        # Calculate sell accuracy (inverse)
        sell_targets = targets['next_up'][sell_signals]
        sell_accuracy = 1 - sell_targets.mean()

        # Overall accuracy
        all_signals = buy_signals | sell_signals
        predictions = np.where(signals[indicator][all_signals] == 1, 1, 0)
        actual = targets['next_up'][all_signals]
        overall_accuracy = accuracy_score(actual, predictions)

        results[indicator] = {
          'buy_accuracy': buy_accuracy,
          'sell_accuracy': sell_accuracy,
          'overall_accuracy': overall_accuracy,
          'buy_signals': buy_signals.sum(),
          'sell_signals': sell_signals.sum()
        }

        print(f"{indicator:12}: Buy {buy_accuracy:.3f} ({buy_signals.sum():3d}) | "
              f"Sell {sell_accuracy:.3f} ({sell_signals.sum():3d}) | "
              f"Overall {overall_accuracy:.3f}")

    except Exception as e:
      continue

  return results


def test_combinations(signals, targets, timeframe_name):
  """Test indicator combinations using RandomForest"""
  indicators = ['MACD', 'Momentum_5', 'Momentum_10', 'WILLR', 'STOCH_D']
  results = {}

  print(f"\n{timeframe_name} Combination Results (RandomForest):")
  print("-" * 50)

  # Prepare data for ML
  valid_mask = ~targets['next_up'].isna()
  X = signals[indicators][valid_mask]
  y = targets['next_up'][valid_mask]

  if len(X) < 50:
    print("Insufficient data for combination analysis")
    return results

  # Test 2-indicator combinations
  best_2_accuracy = 0
  best_2_combo = None
  best_2_rf = None

  for combo in combinations(indicators, 2):
    try:
      X_combo = X[list(combo)]

      # Skip if not enough signals
      signal_mask = (X_combo != 0).any(axis=1)
      if signal_mask.sum() < 30:
        continue

      X_combo_filtered = X_combo[signal_mask]
      y_combo_filtered = y[signal_mask]

      # Train-test split
      X_train, X_test, y_train, y_test = train_test_split(
        X_combo_filtered, y_combo_filtered, test_size=0.3, random_state=42, stratify=y_combo_filtered
      )

      # Train RandomForest
      rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced'
      )
      rf.fit(X_train, y_train)

      # Predict and calculate accuracy
      y_pred = rf.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)

      if accuracy > best_2_accuracy:
        best_2_accuracy = accuracy
        best_2_combo = combo
        best_2_rf = rf

    except Exception as e:
      continue

  # Test 3-indicator combinations (sample 15 random ones)
  best_3_accuracy = 0
  best_3_combo = None
  best_3_rf = None

  import random
  random.seed(42)
  all_3_combos = list(combinations(indicators, 3))
  test_3_combos = random.sample(all_3_combos, min(15, len(all_3_combos)))

  for combo in test_3_combos:
    try:
      X_combo = X[list(combo)]

      # Skip if not enough signals
      signal_mask = (X_combo != 0).any(axis=1)
      if signal_mask.sum() < 30:
        continue

      X_combo_filtered = X_combo[signal_mask]
      y_combo_filtered = y[signal_mask]

      # Train-test split
      X_train, X_test, y_train, y_test = train_test_split(
        X_combo_filtered, y_combo_filtered, test_size=0.3, random_state=42, stratify=y_combo_filtered
      )

      # Train RandomForest
      rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced'
      )
      rf.fit(X_train, y_train)

      # Predict and calculate accuracy
      y_pred = rf.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)

      if accuracy > best_3_accuracy:
        best_3_accuracy = accuracy
        best_3_combo = combo
        best_3_rf = rf

    except Exception as e:
      continue

  # Store and print results
  if best_2_combo:
    results['best_2_combo'] = {
      'indicators': best_2_combo,
      'accuracy': best_2_accuracy,
      'model': best_2_rf
    }

    # Get feature importance
    importance = dict(zip(best_2_combo, best_2_rf.feature_importances_))
    importance_str = " | ".join([f"{ind}: {imp:.3f}" for ind, imp in importance.items()])

    print(f"Best 2-combo: {best_2_combo} -> {best_2_accuracy:.3f}")
    print(f"  Feature importance: {importance_str}")

  if best_3_combo:
    results['best_3_combo'] = {
      'indicators': best_3_combo,
      'accuracy': best_3_accuracy,
      'model': best_3_rf
    }

    # Get feature importance
    importance = dict(zip(best_3_combo, best_3_rf.feature_importances_))
    importance_str = " | ".join([f"{ind}: {imp:.3f}" for ind, imp in importance.items()])

    print(f"Best 3-combo: {best_3_combo} -> {best_3_accuracy:.3f}")
    print(f"  Feature importance: {importance_str}")

  return results


def find_collision_signals(signals_dict, collision_window_hours=4):
  """Find signals that collide across timeframes using RandomForest predictions"""

  # Generate RandomForest signals for each timeframe
  rf_signals = {}

  for tf_name in ['4h', '6h', '8h']:
    try:
      signals = signals_dict[tf_name]['signals']
      targets = signals_dict[tf_name]['targets']

      # Use all 5 indicators for RandomForest
      indicators = ['MACD', 'Momentum_5', 'Momentum_10', 'WILLR', 'STOCH_D']

      # Prepare data
      valid_mask = ~targets['next_up'].isna()
      X = signals[indicators][valid_mask]
      y = targets['next_up'][valid_mask]

      # Filter to rows with at least one signal
      signal_mask = (X != 0).any(axis=1)
      X_filtered = X[signal_mask]
      y_filtered = y[signal_mask]

      if len(X_filtered) >= 30:
        # Train RandomForest
        rf = RandomForestClassifier(
          n_estimators=150,
          random_state=42,
          max_depth=6,
          min_samples_split=8,
          class_weight='balanced'
        )
        rf.fit(X_filtered, y_filtered)

        # Generate predictions for all data where we have signals
        rf_predictions = pd.Series(index=signals.index, dtype=int)
        rf_predictions.loc[signal_mask[signal_mask].index] = rf.predict(X_filtered)

        # Convert to buy/sell signals (1 for buy prediction, -1 for sell prediction)
        rf_signals[tf_name] = np.where(rf_predictions == 1, 1,
                                       np.where(rf_predictions == 0, -1, 0))
        rf_signals[tf_name] = pd.Series(rf_signals[tf_name], index=signals.index)
      else:
        rf_signals[tf_name] = pd.Series(0, index=signals.index)

    except Exception as e:
      rf_signals[tf_name] = pd.Series(0, index=signals_dict[tf_name]['signals'].index)

  # Find collisions between RandomForest signals
  base_index = rf_signals['4h'].index
  collision_results = pd.DataFrame(index=base_index)

  collision_results['4h_rf_signal'] = rf_signals['4h']
  collision_results['6h_rf_signal'] = 0
  collision_results['8h_rf_signal'] = 0
  collision_results['collision_count'] = 0
  collision_results['collision_signal'] = 0

  # For each 4h timestamp, check for collisions
  for timestamp in base_index:
    signal_4h = rf_signals['4h'].loc[timestamp]

    if signal_4h != 0:
      # Find signals in other timeframes within collision window
      start_window = timestamp - pd.Timedelta(hours=collision_window_hours // 2)
      end_window = timestamp + pd.Timedelta(hours=collision_window_hours // 2)

      # Get 6h signals in window
      signals_6h_in_window = rf_signals['6h'][
        (rf_signals['6h'].index >= start_window) &
        (rf_signals['6h'].index <= end_window) &
        (rf_signals['6h'] != 0)
        ]

      # Get 8h signals in window
      signals_8h_in_window = rf_signals['8h'][
        (rf_signals['8h'].index >= start_window) &
        (rf_signals['8h'].index <= end_window) &
        (rf_signals['8h'] != 0)
        ]

      # Determine dominant signals in window
      signal_6h = signals_6h_in_window.mode().iloc[0] if len(signals_6h_in_window) > 0 else 0
      signal_8h = signals_8h_in_window.mode().iloc[0] if len(signals_8h_in_window) > 0 else 0

      collision_results.loc[timestamp, '6h_rf_signal'] = signal_6h
      collision_results.loc[timestamp, '8h_rf_signal'] = signal_8h

      # Count collisions (how many timeframes agree)
      signals = [signal_4h, signal_6h, signal_8h]
      non_zero_signals = [s for s in signals if s != 0]

      if len(non_zero_signals) >= 2:
        # Check if they agree in direction
        if all(s > 0 for s in non_zero_signals) or all(s < 0 for s in non_zero_signals):
          collision_results.loc[timestamp, 'collision_count'] = len(non_zero_signals)
          collision_results.loc[timestamp, 'collision_signal'] = non_zero_signals[0]

  return collision_results


def analyze_collisions(collision_results, targets_4h):
  """Analyze accuracy of RandomForest collision signals"""
  print(f"\nCross-Timeframe RandomForest Collision Analysis:")
  print("-" * 50)

  # Filter to collision signals only
  collision_mask = collision_results['collision_count'] >= 2

  if collision_mask.sum() < 5:
    print("Insufficient collision signals for analysis")
    return {}

  # Calculate collision accuracy
  collision_predictions = (collision_results['collision_signal'][collision_mask] > 0).astype(int)
  collision_targets = targets_4h['next_up'][collision_mask]

  # Remove NaN values
  valid_mask = ~collision_targets.isna()
  if valid_mask.sum() < 3:
    print("Insufficient valid collision signals")
    return {}

  collision_predictions = collision_predictions[valid_mask]
  collision_targets = collision_targets[valid_mask]

  if len(set(collision_predictions)) > 1:
    collision_accuracy = accuracy_score(collision_targets, collision_predictions)
    collision_precision = precision_score(collision_targets, collision_predictions, average='weighted')
    collision_recall = recall_score(collision_targets, collision_predictions, average='weighted')
  else:
    collision_accuracy = collision_targets.mean() if collision_predictions[0] == 1 else 1 - collision_targets.mean()
    collision_precision = collision_accuracy
    collision_recall = collision_accuracy

  # Break down by collision count
  for count in [2, 3]:
    count_mask = collision_results['collision_count'] == count
    if count_mask.sum() > 2:
      count_predictions = (collision_results['collision_signal'][count_mask] > 0).astype(int)
      count_targets = targets_4h['next_up'][count_mask]
      count_valid = ~count_targets.isna()

      if count_valid.sum() > 2:
        if len(set(count_predictions[count_valid])) > 1:
          count_accuracy = accuracy_score(count_targets[count_valid], count_predictions[count_valid])
        else:
          count_targets_valid = count_targets[count_valid]
          count_accuracy = count_targets_valid.mean() if count_predictions[count_valid].iloc[
                                                           0] == 1 else 1 - count_targets_valid.mean()

        print(f"{count} timeframes agree: {count_accuracy:.3f} accuracy ({count_valid.sum()} signals)")

  print(f"Overall collision accuracy: {collision_accuracy:.3f} ({len(collision_predictions)} signals)")
  print(f"Collision precision: {collision_precision:.3f}")
  print(f"Collision recall: {collision_recall:.3f}")
  print(f"Collision rate: {collision_mask.sum() / len(collision_results):.3f}")

  return {
    'accuracy': collision_accuracy,
    'precision': collision_precision,
    'recall': collision_recall,
    'signal_count': len(collision_predictions),
    'collision_rate': collision_mask.sum() / len(collision_results)
  }


def main_analysis(file_path):
  """Main analysis function"""
  print(f"Loading 1h data from: {file_path}")

  # Load 1h data
  df_1h = load_and_prepare_data(file_path)
  print(f"Loaded {len(df_1h)} 1h records from {df_1h.index.min()} to {df_1h.index.max()}")

  # Create timeframes
  print("Creating 4h, 6h, and 8h timeframes...")
  timeframes = create_timeframes(df_1h)

  # Analyze each timeframe
  all_results = {}
  signals_dict = {}

  for tf_name, df in timeframes.items():
    print(f"\n{'=' * 60}")
    print(f"ANALYZING {tf_name.upper()} TIMEFRAME ({len(df)} records)")
    print('=' * 60)

    # Calculate indicators and signals
    data_with_indicators = calculate_indicators(df)
    signals = generate_signals(data_with_indicators)
    targets = create_targets(data_with_indicators)

    # Store for collision analysis
    signals_dict[tf_name] = {
      'signals': signals,
      'targets': targets,
      'data': data_with_indicators
    }

    # Test individual indicators
    individual_results = test_individual_indicators(signals, targets, tf_name)

    # Test combinations
    combination_results = test_combinations(signals, targets, tf_name)

    all_results[tf_name] = {
      'individual': individual_results,
      'combinations': combination_results
    }

  # Cross-timeframe collision analysis
  print(f"\n{'=' * 60}")
  print("CROSS-TIMEFRAME COLLISION ANALYSIS")
  print('=' * 60)

  collision_results = find_collision_signals(signals_dict, collision_window_hours=4)
  collision_analysis = analyze_collisions(collision_results, signals_dict['4h']['targets'])

  return {
    'timeframe_results': all_results,
    'collision_analysis': collision_analysis,
    'signals_dict': signals_dict,
    'collision_results': collision_results
  }


# Example usage
if __name__ == "__main__":
  file_path = "./datasets/ETHUSDT_1h.csv"

  try:
    results = main_analysis(file_path)
    print("\nAnalysis completed successfully!")

  except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please ensure you have a 1h timeframe CSV file")
  except Exception as e:
    print(f"Error in analysis: {e}")
    import traceback

    traceback.print_exc()