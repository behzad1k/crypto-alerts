import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
  """Load ETH data and prepare for analysis"""
  df = pd.read_csv(file_path)
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date').reset_index(drop=True)
  return df


def calculate_technical_indicators(df):
  """Calculate technical indicators (reusing from previous analysis)"""
  data = df.copy()

  # Simple Moving Averages
  data['SMA_5'] = data['Close'].rolling(window=5).mean()
  data['SMA_20'] = data['Close'].rolling(window=20).mean()
  data['SMA_50'] = data['Close'].rolling(window=50).mean()

  # Exponential Moving Averages
  data['EMA_12'] = data['Close'].ewm(span=12).mean()
  data['EMA_26'] = data['Close'].ewm(span=26).mean()
  data['EMA_50'] = data['Close'].ewm(span=50).mean()

  # RSI
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

  # ATR
  high_low = data['High'] - data['Low']
  high_close = np.abs(data['High'] - data['Close'].shift())
  low_close = np.abs(data['Low'] - data['Close'].shift())
  tr = np.maximum(high_low, np.maximum(high_close, low_close))
  data['ATR'] = tr.rolling(window=14).mean()

  # Volume indicators
  data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
  data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']

  # Momentum
  data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
  data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

  # Money Flow Index (MFI)
  typical_price = (data['High'] + data['Low'] + data['Close']) / 3
  raw_money_flow = typical_price * data['Volume']
  positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
  negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
  positive_flow_sum = positive_flow.rolling(window=14).sum()
  negative_flow_sum = negative_flow.rolling(window=14).sum()
  mfr = positive_flow_sum / negative_flow_sum
  data['MFI'] = 100 - (100 / (1 + mfr))

  return data


class TradingStrategy:
  def __init__(self, initial_capital=1000, risk_per_trade=0.02, stop_loss_pct=0.05):
    self.initial_capital = initial_capital
    self.capital = initial_capital
    self.position = 0  # 0 = no position, 1 = long, -1 = short
    self.entry_price = 0
    self.eth_holdings = 0
    self.risk_per_trade = risk_per_trade  # Risk 2% of capital per trade
    self.stop_loss_pct = stop_loss_pct  # 5% stop loss

    # Tracking
    self.trades = []
    self.portfolio_values = []
    self.signals = []
    self.max_drawdown = 0
    self.peak_value = initial_capital

  def calculate_position_size(self, signal_confidence, current_price):
    """Calculate position size based on signal confidence and risk management"""
    if signal_confidence == 'high':
      risk_multiplier = 3.0  # Use 3x normal risk for high confidence
    elif signal_confidence == 'medium':
      risk_multiplier = 1.5  # Use 1.5x normal risk for medium confidence
    else:
      risk_multiplier = 1.0  # Normal risk for low confidence

    # Calculate position size based on stop loss
    risk_amount = self.capital * self.risk_per_trade * risk_multiplier
    position_size = risk_amount / (current_price * self.stop_loss_pct)

    # Don't risk more than 50% of capital in a single trade
    max_position_value = self.capital * 0.5
    max_position_size = max_position_value / current_price

    return min(position_size, max_position_size)

  def get_signal_confidence(self, row):
    """Determine signal confidence based on our analysis insights"""
    signals = []
    confidence_scores = []

    # High-confidence 1-day signal (91.4% accuracy): MACD_signal + Momentum_10
    if not pd.isna(row['MACD_signal']) and not pd.isna(row['Momentum_10']):
      if row['MACD_signal'] > 0 and row['Momentum_10'] > 0.02:  # Strong positive momentum
        signals.append('long')
        confidence_scores.append('high')
      elif row['MACD_signal'] < 0 and row['Momentum_10'] < -0.02:  # Strong negative momentum
        signals.append('short')
        confidence_scores.append('high')

    # High-confidence 3-day signal (80.4% accuracy): MACD_signal + MACD + MFI
    if not pd.isna(row['MACD_signal']) and not pd.isna(row['MACD']) and not pd.isna(row['MFI']):
      macd_bullish = row['MACD_signal'] > 0 and row['MACD'] > 0
      mfi_oversold = row['MFI'] < 30
      mfi_overbought = row['MFI'] > 70

      if macd_bullish and mfi_oversold:
        signals.append('long')
        confidence_scores.append('high')
      elif not macd_bullish and mfi_overbought:
        signals.append('short')
        confidence_scores.append('high')

    # Medium-confidence volume-based signal (55% accuracy): Volume_SMA + BB_width + ATR
    if not pd.isna(row['Volume_ratio']) and not pd.isna(row['BB_width']) and not pd.isna(row['ATR']):
      volume_surge = row['Volume_ratio'] > 1.5  # Volume 50% above average
      volatility_expanding = row['BB_width'] > row.get('BB_width_SMA', row['BB_width'])

      if volume_surge and volatility_expanding:
        # Use RSI to determine direction for medium confidence signals
        if not pd.isna(row['RSI']):
          if row['RSI'] < 40:  # Oversold, expect bounce
            signals.append('long')
            confidence_scores.append('medium')
          elif row['RSI'] > 60:  # Overbought, expect pullback
            signals.append('short')
            confidence_scores.append('medium')

    # Return most confident signal
    if 'high' in confidence_scores:
      idx = confidence_scores.index('high')
      return signals[idx], 'high'
    elif 'medium' in confidence_scores:
      idx = confidence_scores.index('medium')
      return signals[idx], 'medium'
    else:
      return 'hold', 'low'

  def execute_trade(self, signal, confidence, current_price, date):
    """Execute trading decision"""
    # Close existing position if signal changes
    if self.position != 0:
      # Check stop loss
      if self.position == 1 and current_price <= self.entry_price * (1 - self.stop_loss_pct):
        self.close_position(current_price, date, 'stop_loss')
      elif self.position == -1 and current_price >= self.entry_price * (1 + self.stop_loss_pct):
        self.close_position(current_price, date, 'stop_loss')
      # Check signal reversal
      elif (self.position == 1 and signal == 'short') or (self.position == -1 and signal == 'long'):
        self.close_position(current_price, date, 'signal_reversal')

    # Open new position
    if self.position == 0 and signal in ['long', 'short']:
      position_size = self.calculate_position_size(confidence, current_price)

      if signal == 'long':
        # Buy ETH
        cost = position_size * current_price
        if cost <= self.capital * 0.98:  # Leave 2% for fees/slippage
          self.eth_holdings = position_size
          self.capital -= cost
          self.position = 1
          self.entry_price = current_price

          self.trades.append({
            'date': date,
            'action': 'buy',
            'price': current_price,
            'size': position_size,
            'cost': cost,
            'confidence': confidence,
            'capital_after': self.capital
          })

      # Note: For simplicity, this version only implements long positions
      # Short positions would require margin trading simulation

  def close_position(self, current_price, date, reason):
    """Close current position"""
    if self.position == 1 and self.eth_holdings > 0:
      # Sell ETH
      proceeds = self.eth_holdings * current_price
      pnl = proceeds - (self.eth_holdings * self.entry_price)

      self.capital += proceeds

      self.trades.append({
        'date': date,
        'action': 'sell',
        'price': current_price,
        'size': self.eth_holdings,
        'proceeds': proceeds,
        'pnl': pnl,
        'reason': reason,
        'capital_after': self.capital
      })

      self.eth_holdings = 0
      self.position = 0
      self.entry_price = 0

  def update_portfolio_value(self, current_price, date):
    """Update portfolio value"""
    if self.position == 1:
      portfolio_value = self.capital + (self.eth_holdings * current_price)
    else:
      portfolio_value = self.capital

    self.portfolio_values.append({
      'date': date,
      'portfolio_value': portfolio_value,
      'cash': self.capital,
      'eth_value': self.eth_holdings * current_price if self.position == 1 else 0,
      'eth_holdings': self.eth_holdings
    })

    # Track drawdown
    if portfolio_value > self.peak_value:
      self.peak_value = portfolio_value

    drawdown = (self.peak_value - portfolio_value) / self.peak_value
    if drawdown > self.max_drawdown:
      self.max_drawdown = drawdown


def run_backtest(file_path, initial_capital=1000):
  """Run the complete backtesting simulation"""

  print("Loading and preparing data...")
  df = load_and_prepare_data(file_path)

  print("Calculating technical indicators...")
  data = calculate_technical_indicators(df)

  print("Running trading simulation...")
  strategy = TradingStrategy(initial_capital=initial_capital)

  # Skip first 50 rows to allow indicators to stabilize
  for i in range(50, len(data)):
    row = data.iloc[i]
    date = row['Date']
    current_price = row['Close']

    # Get trading signal
    signal, confidence = strategy.get_signal_confidence(row)

    # Execute trade
    strategy.execute_trade(signal, confidence, current_price, date)

    # Update portfolio tracking
    strategy.update_portfolio_value(current_price, date)

    # Store signal for analysis
    strategy.signals.append({
      'date': date,
      'price': current_price,
      'signal': signal,
      'confidence': confidence
    })

  # Close any remaining position
  if strategy.position != 0:
    final_price = data.iloc[-1]['Close']
    final_date = data.iloc[-1]['Date']
    strategy.close_position(final_price, final_date, 'end_of_data')
    strategy.update_portfolio_value(final_price, final_date)

  return strategy, data


def analyze_results(strategy, data):
  """Analyze and display backtesting results"""

  portfolio_df = pd.DataFrame(strategy.portfolio_values)
  trades_df = pd.DataFrame(strategy.trades) if strategy.trades else pd.DataFrame()
  signals_df = pd.DataFrame(strategy.signals)

  # Calculate key metrics
  final_value = portfolio_df['portfolio_value'].iloc[-1]
  total_return = (final_value / strategy.initial_capital - 1) * 100

  # Calculate buy and hold return for comparison
  initial_price = data['Close'].iloc[50]  # Start from same point as strategy
  final_price = data['Close'].iloc[-1]
  buy_hold_return = (final_price / initial_price - 1) * 100

  # Calculate annualized returns
  trading_days = len(portfolio_df)
  years = trading_days / (365 * 24)  # Hourly data
  annual_return = (final_value / strategy.initial_capital) ** (1 / years) - 1

  # Trading statistics
  if not trades_df.empty:
    buy_trades = trades_df[trades_df['action'] == 'buy']
    sell_trades = trades_df[trades_df['action'] == 'sell']

    if not sell_trades.empty:
      winning_trades = sell_trades[sell_trades['pnl'] > 0]
      win_rate = len(winning_trades) / len(sell_trades) * 100

      avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
      avg_loss = sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()
      avg_loss = avg_loss if not pd.isna(avg_loss) else 0
    else:
      win_rate = 0
      avg_win = 0
      avg_loss = 0
  else:
    win_rate = 0
    avg_win = 0
    avg_loss = 0

  # Print results
  print("\n" + "=" * 60)
  print("BACKTESTING RESULTS")
  print("=" * 60)
  print(f"Initial Capital:        ${strategy.initial_capital:,.2f}")
  print(f"Final Portfolio Value:  ${final_value:,.2f}")
  print(f"Total Return:          {total_return:+.2f}%")
  print(f"Buy & Hold Return:     {buy_hold_return:+.2f}%")
  print(f"Outperformance:        {total_return - buy_hold_return:+.2f}%")
  print(f"Annualized Return:     {annual_return * 100:.2f}%")
  print(f"Max Drawdown:          {strategy.max_drawdown * 100:.2f}%")
  print(f"\nTrading Statistics:")
  print(f"Total Trades:          {len(trades_df)}")
  print(f"Win Rate:              {win_rate:.1f}%")
  print(f"Average Win:           ${avg_win:.2f}")
  print(f"Average Loss:          ${avg_loss:.2f}")

  # Signal statistics
  signal_counts = signals_df['signal'].value_counts()
  confidence_counts = signals_df['confidence'].value_counts()

  print(f"\nSignal Statistics:")
  print(f"Long signals:          {signal_counts.get('long', 0)}")
  print(f"Short signals:         {signal_counts.get('short', 0)}")
  print(f"Hold signals:          {signal_counts.get('hold', 0)}")
  print(f"High confidence:       {confidence_counts.get('high', 0)}")
  print(f"Medium confidence:     {confidence_counts.get('medium', 0)}")
  print(f"Low confidence:        {confidence_counts.get('low', 0)}")

  return portfolio_df, trades_df, signals_df


def plot_results(portfolio_df, data, trades_df):
  """Create visualization of results"""

  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

  # Plot 1: Portfolio value vs ETH price
  ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'],
           label='Portfolio Value', linewidth=2, color='green')

  # Add buy/sell markers
  if not trades_df.empty:
    buy_trades = trades_df[trades_df['action'] == 'buy']
    sell_trades = trades_df[trades_df['action'] == 'sell']

    if not buy_trades.empty:
      ax1.scatter(buy_trades['date'],
                  [portfolio_df[portfolio_df['date'] == d]['portfolio_value'].iloc[0]
                   for d in buy_trades['date']],
                  color='blue', marker='^', s=100, label='Buy', zorder=5)

    if not sell_trades.empty:
      ax1.scatter(sell_trades['date'],
                  [portfolio_df[portfolio_df['date'] == d]['portfolio_value'].iloc[0]
                   for d in sell_trades['date']],
                  color='red', marker='v', s=100, label='Sell', zorder=5)

  ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
  ax1.set_ylabel('Portfolio Value ($)')
  ax1.set_title('Portfolio Performance')
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  # Plot 2: ETH price with buy/sell signals
  eth_data = data[data['Date'].isin(portfolio_df['date'])]
  ax2.plot(eth_data['Date'], eth_data['Close'], label='ETH Price', color='orange', alpha=0.8)

  if not trades_df.empty:
    if not buy_trades.empty:
      ax2.scatter(buy_trades['date'], buy_trades['price'],
                  color='blue', marker='^', s=100, label='Buy Signal', zorder=5)
    if not sell_trades.empty:
      ax2.scatter(sell_trades['date'], sell_trades['price'],
                  color='red', marker='v', s=100, label='Sell Signal', zorder=5)

  ax2.set_ylabel('ETH Price ($)')
  ax2.set_title('ETH Price with Trading Signals')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  # Plot 3: Portfolio composition over time
  ax3.fill_between(portfolio_df['date'], 0, portfolio_df['cash'],
                   label='Cash', alpha=0.7, color='lightblue')
  ax3.fill_between(portfolio_df['date'], portfolio_df['cash'],
                   portfolio_df['portfolio_value'],
                   label='ETH Holdings', alpha=0.7, color='orange')

  ax3.set_ylabel('Value ($)')
  ax3.set_xlabel('Date')
  ax3.set_title('Portfolio Composition Over Time')
  ax3.legend()
  ax3.grid(True, alpha=0.3)

  # Format x-axis
  for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

  plt.tight_layout()
  plt.show()


# Main execution
if __name__ == "__main__":
  # Set your CSV file path here
  file_path = "ETH.csv"

  try:
    # Run backtest
    strategy, data = run_backtest(file_path, initial_capital=1000)

    # Analyze results
    portfolio_df, trades_df, signals_df = analyze_results(strategy, data)

    # Create visualizations
    plot_results(portfolio_df, data, trades_df)

    # Display recent trades
    if not trades_df.empty:
      print(f"\nRecent Trades:")
      print(trades_df[['date', 'action', 'price', 'confidence', 'pnl']].tail(10))

  except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please update the file_path variable with your CSV file location")
  except Exception as e:
    print(f"Error in simulation: {e}")
    import traceback

    traceback.print_exc()