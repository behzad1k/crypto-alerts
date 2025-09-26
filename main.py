import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
  """Load ETH data and prepare for analysis"""
  df = pd.read_csv(file_path)
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date').reset_index(drop=True)
  return df


def calculate_technical_indicators(df):
  """Calculate various technical indicators"""
  data = df.copy()

  # Simple Moving Averages
  data['SMA_5'] = data['Close'].rolling(window=5).mean()
  data['SMA_10'] = data['Close'].rolling(window=10).mean()
  data['SMA_20'] = data['Close'].rolling(window=20).mean()
  data['SMA_50'] = data['Close'].rolling(window=50).mean()

  # Exponential Moving Averages
  data['EMA_12'] = data['Close'].ewm(span=12).mean()
  data['EMA_26'] = data['Close'].ewm(span=26).mean()
  data['EMA_50'] = data['Close'].ewm(span=50).mean()

  # RSI (Relative Strength Index)
  delta = data['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  data['RSI'] = 100 - (100 / (1 + rs))

  # MACD
  data['MACD'] = data['EMA_12'] - data['EMA_26']
  data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
  data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

  # Bollinger Bands
  data['BB_middle'] = data['Close'].rolling(window=20).mean()
  bb_std = data['Close'].rolling(window=20).std()
  data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
  data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
  data['BB_width'] = data['BB_upper'] - data['BB_lower']
  data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])

  # Stochastic Oscillator
  low_min = data['Low'].rolling(window=14).min()
  high_max = data['High'].rolling(window=14).max()
  data['Stoch_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
  data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

  # Williams %R
  data['Williams_R'] = -100 * ((high_max - data['Close']) / (high_max - low_min))

  # Commodity Channel Index (CCI)
  tp = (data['High'] + data['Low'] + data['Close']) / 3
  sma_tp = tp.rolling(window=20).mean()
  mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
  data['CCI'] = (tp - sma_tp) / (0.015 * mad)

  # Average True Range (ATR)
  high_low = data['High'] - data['Low']
  high_close = np.abs(data['High'] - data['Close'].shift())
  low_close = np.abs(data['Low'] - data['Close'].shift())
  tr = np.maximum(high_low, np.maximum(high_close, low_close))
  data['ATR'] = tr.rolling(window=14).mean()

  # Volume indicators
  data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
  data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']

  # Price momentum
  data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
  data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
  data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

  # Rate of Change (ROC)
  data['ROC_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
  data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

  # Money Flow Index (MFI)
  typical_price = (data['High'] + data['Low'] + data['Close']) / 3
  raw_money_flow = typical_price * data['Volume']
  positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
  negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
  positive_flow_sum = positive_flow.rolling(window=14).sum()
  negative_flow_sum = negative_flow.rolling(window=14).sum()
  mfr = positive_flow_sum / negative_flow_sum
  data['MFI'] = 100 - (100 / (1 + mfr))

  # Crossover signals
  data['SMA_5_20_cross'] = np.where(data['SMA_5'] > data['SMA_20'], 1, 0)
  data['EMA_12_26_cross'] = np.where(data['EMA_12'] > data['EMA_26'], 1, 0)
  data['MACD_signal_cross'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)

  return data


def create_target_variables(data, periods=[1, 3, 5, 10]):
  """Create target variables for different prediction horizons"""
  targets = {}

  for period in periods:
    # Binary classification: will price go up in next N periods?
    future_return = (data['Close'].shift(-period) / data['Close'] - 1)
    targets[f'target_{period}d_up'] = (future_return > 0).astype(int)

    # Multi-class: significant moves (>2% up, >2% down, sideways)
    targets[f'target_{period}d_class'] = np.where(
      future_return > 0.02, 2,  # Strong up
      np.where(future_return < -0.02, 0, 1)  # Strong down, sideways
    )

  return pd.DataFrame(targets, index=data.index)


def get_indicator_features(data):
  """Get list of calculated indicator columns"""
  exclude_cols = ['Unix Timestamp', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
  indicator_cols = [col for col in data.columns if col not in exclude_cols]
  return indicator_cols


def test_individual_indicators(data, targets, indicator_cols):
  """Test accuracy of individual indicators"""
  results = {}

  for target_col in targets.columns:
    print(f"\nTesting indicators for {target_col}:")
    target_results = {}

    # Remove rows with NaN values for this target
    valid_mask = ~(data[indicator_cols].isna().any(axis=1) | targets[target_col].isna())

    for indicator in indicator_cols:
      try:
        # Skip if indicator has too many NaN values
        if data[indicator].isna().sum() > len(data) * 0.3:
          continue

        # Create simple signal based on indicator
        if 'cross' in indicator.lower():
          # Already binary
          signals = data[indicator][valid_mask]
        elif 'rsi' in indicator.lower():
          # RSI oversold/overbought signals
          signals = ((data[indicator] < 30) | (data[indicator] > 70))[valid_mask]
        elif 'stoch' in indicator.lower():
          # Stochastic oversold/overbought
          signals = ((data[indicator] < 20) | (data[indicator] > 80))[valid_mask]
        elif 'williams' in indicator.lower():
          # Williams %R signals
          signals = ((data[indicator] < -80) | (data[indicator] > -20))[valid_mask]
        elif 'bb_position' in indicator.lower():
          # Bollinger Band position signals
          signals = ((data[indicator] < 0.1) | (data[indicator] > 0.9))[valid_mask]
        elif 'macd' in indicator.lower() and 'cross' not in indicator.lower():
          # MACD above/below zero
          signals = (data[indicator] > 0)[valid_mask]
        else:
          # For other indicators, use above/below median as signal
          median_val = data[indicator].median()
          signals = (data[indicator] > median_val)[valid_mask]

        # Calculate accuracy
        if len(signals) > 0 and len(set(signals)) > 1:
          accuracy = accuracy_score(targets[target_col][valid_mask], signals)
          target_results[indicator] = accuracy

      except Exception as e:
        continue

    # Sort by accuracy
    sorted_results = sorted(target_results.items(), key=lambda x: x[1], reverse=True)
    results[target_col] = sorted_results

    # Print top 10
    print(f"Top 10 indicators for {target_col}:")
    for i, (indicator, accuracy) in enumerate(sorted_results[:10], 1):
      print(f"{i:2d}. {indicator:<25}: {accuracy:.4f}")

  return results


def test_indicator_combinations(data, targets, top_indicators, max_combinations=3):
  """Test combinations of top indicators using Random Forest"""
  results = {}

  for target_col in targets.columns:
    print(f"\nTesting combinations for {target_col}:")

    # Get valid data
    all_indicators = [ind[0] for ind in top_indicators[target_col][:15]]  # Top 15 indicators
    valid_mask = ~(data[all_indicators].isna().any(axis=1) | targets[target_col].isna())

    X = data[all_indicators][valid_mask]
    y = targets[target_col][valid_mask]

    if len(X) < 100:  # Need sufficient data
      continue

    combination_results = {}

    # Test different combination sizes
    for combo_size in range(2, min(max_combinations + 1, len(all_indicators) + 1)):
      print(f"  Testing {combo_size}-indicator combinations...")
      best_combo_score = 0
      best_combo = None

      # Test combinations of top indicators
      for combo in combinations(all_indicators[:10], combo_size):
        try:
          # Use Random Forest for combination
          X_combo = X[list(combo)]

          # Split data
          X_train, X_test, y_train, y_test = train_test_split(
            X_combo, y, test_size=0.3, random_state=42, stratify=y
          )

          # Train Random Forest
          rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_split=10
          )
          rf.fit(X_train, y_train)

          # Predict and calculate accuracy
          y_pred = rf.predict(X_test)
          accuracy = accuracy_score(y_test, y_pred)

          if accuracy > best_combo_score:
            best_combo_score = accuracy
            best_combo = combo

        except Exception as e:
          continue

      if best_combo:
        combination_results[f'{combo_size}_indicators'] = {
          'indicators': best_combo,
          'accuracy': best_combo_score
        }
        print(f"    Best {combo_size}-combo: {best_combo} -> {best_combo_score:.4f}")

    results[target_col] = combination_results

  return results


def analyze_feature_importance(data, targets, top_indicators):
  """Analyze feature importance using Random Forest"""
  importance_results = {}

  for target_col in targets.columns:
    print(f"\nFeature importance analysis for {target_col}:")

    # Get top 10 indicators
    top_10_indicators = [ind[0] for ind in top_indicators[target_col][:10]]
    valid_mask = ~(data[top_10_indicators].isna().any(axis=1) | targets[target_col].isna())

    X = data[top_10_indicators][valid_mask]
    y = targets[target_col][valid_mask]

    if len(X) < 100:
      continue

    try:
      # Train Random Forest
      rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
      rf.fit(X, y)

      # Get feature importance
      importance_df = pd.DataFrame({
        'indicator': X.columns,
        'importance': rf.feature_importances_
      }).sort_values('importance', ascending=False)

      importance_results[target_col] = importance_df

      print("Feature Importance Rankings:")
      for i, row in importance_df.iterrows():
        print(f"  {row['indicator']:<25}: {row['importance']:.4f}")

    except Exception as e:
      print(f"  Error in feature importance analysis: {e}")

  return importance_results


def main_analysis(file_path):
  """Main analysis function"""
  print("Loading and preparing ETH data...")

  # Load data
  df = load_and_prepare_data(file_path)
  print(f"Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

  # Calculate indicators
  print("Calculating technical indicators...")
  data_with_indicators = calculate_technical_indicators(df)

  # Create targets
  print("Creating target variables...")
  targets = create_target_variables(data_with_indicators)

  # Get indicator columns
  indicator_cols = get_indicator_features(data_with_indicators)
  print(f"Calculated {len(indicator_cols)} technical indicators")

  # Test individual indicators
  print("\n" + "=" * 60)
  print("TESTING INDIVIDUAL INDICATORS")
  print("=" * 60)
  individual_results = test_individual_indicators(data_with_indicators, targets, indicator_cols)

  # Test combinations
  print("\n" + "=" * 60)
  print("TESTING INDICATOR COMBINATIONS")
  print("=" * 60)
  combination_results = test_indicator_combinations(data_with_indicators, targets, individual_results)

  # Feature importance analysis
  print("\n" + "=" * 60)
  print("FEATURE IMPORTANCE ANALYSIS")
  print("=" * 60)
  importance_results = analyze_feature_importance(data_with_indicators, targets, individual_results)

  return {
    'individual_results': individual_results,
    'combination_results': combination_results,
    'importance_results': importance_results,
    'data': data_with_indicators,
    'targets': targets
  }


# Example usage:
if __name__ == "__main__":
  # Replace with your CSV file path
  file_path = "ETH.csv"

  try:
    results = main_analysis(file_path)
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nKey findings:")
    print("- Check individual_results for single indicator performance")
    print("- Check combination_results for best indicator combinations")
    print("- Check importance_results for Random Forest feature rankings")

  except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please update the file_path variable with your CSV file location")
  except Exception as e:
    print(f"Error in analysis: {e}")