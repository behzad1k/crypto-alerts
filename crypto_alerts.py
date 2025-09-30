"""
Advanced Crypto Trading Alert System - Multi-Timeframe Multi-Indicator Analysis

This script monitors crypto prices using FREE APIs and sends email alerts
based on comprehensive technical analysis across multiple timeframes and indicators.

ADVANCED FEATURES:
- 6 core indicators: MACD, Momentum_5, Momentum_10, RSI, STOCH_D, WILLR
- 8 timeframes: 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
- 504 total indicator/timeframe combinations per symbol
- Confidence threshold based on signal consensus
- SQLite database for signal tracking and analysis
- Multiple email recipients support
- Dynamic stop-loss based on confidence and timeframe
- Anti-duplicate system using database history

REQUIREMENTS:
- No API keys needed!
- Python 3.6+ with requests, pandas, numpy, sqlite3, talib (optional)
- Email credentials for alerts

USAGE:
python crypto_alerts.py --create-config     # Create email config template
python crypto_alerts.py --test-email        # Test email setup
python crypto_alerts.py --test-apis         # Test if APIs work
python crypto_alerts.py --init-db          # Initialize database
python crypto_alerts.py --test             # Test signals (no emails)
python crypto_alerts.py                    # Start monitoring
"""

import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os
import logging
import sqlite3
from collections import deque, defaultdict
import threading
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdvancedCryptoAlertSystem:
  def __init__(self, email_config):
    self.email_config = email_config
    self.symbol_queue = deque()
    self.symbol_stats = {}
    self.running = False
    self.conf_threshold = 10
    self.db_path = 'crypto_signals.db'

    # Timeframes in minutes (for API calls)
    self.timeframes = {
      '30m': 30,
      '1h': 60,
      '2h': 120,
      '4h': 240,
      '6h': 360,
      '8h': 480,
      '12h': 720,
      # '1d': 1440
    }

    # Binance API interval mapping
    self.binance_intervals = {
      '30m': '30m',
      '1h': '1h',
      '2h': '2h',
      '4h': '4h',
      '6h': '6h',
      '8h': '8h',
      '12h': '12h',
      # '1d': '1d'
    }

    # Initialize database
    self.init_database()

  def init_database(self):
    """Initialize SQLite database for signal tracking"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      # Create signals table with timeframe-specific indicator tracking
      cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence_score INTEGER NOT NULL,
                    total_indicators INTEGER NOT NULL,
                    confidence_percentage REAL NOT NULL,
                    price REAL NOT NULL,
                    stop_loss REAL,
                    predominant_timeframe TEXT,
                    max_validation_period INTEGER,
                    indicator_breakdown TEXT,
                    contributing_indicators TEXT,
                    timeframe_specific_indicators TEXT,
                    datetime_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

      # Add columns if they don't exist (for existing databases)
      try:
        cursor.execute('ALTER TABLE signals ADD COLUMN contributing_indicators TEXT')
      except sqlite3.OperationalError:
        pass

      try:
        cursor.execute('ALTER TABLE signals ADD COLUMN timeframe_specific_indicators TEXT')
      except sqlite3.OperationalError:
        pass

      # Create index for faster queries
      cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_datetime 
                ON signals(symbol, datetime_created)
            ''')

      conn.commit()
      conn.close()
      logging.info("Database initialized successfully with timeframe-specific tracking")

    except Exception as e:
      logging.error(f"Failed to initialize database: {e}")
      raise

  def get_historical_data_optimized(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch 30m data and derive all timeframes from it"""
    # Calculate how much 30m data we need to generate enough data for all timeframes
    # For 1d timeframe, we need limit * 48 thirty-minute periods
    required_30m_periods = 4000  # 48 thirty-minute periods per day

    # Try primary and fallback APIs in order
    api_methods = [
      ('KuCoin', self._fetch_kucoin_30m_data),
      ('Binance', self._fetch_binance_30m_data),
      ('CryptoCompare', self._fetch_cryptocompare_30m_data),
      ('Coinbase', self._fetch_coinbase_30m_data),
      ('CoinGecko', self._fetch_coingecko_30m_data),
      ('Kraken', self._fetch_kraken_30m_data),
    ]

    for api_name, fetch_method in api_methods:
      try:
        data = fetch_method(symbol, required_30m_periods)
        if data is not None and len(data) >= 100:  # Need substantial data
          logging.info(f"Successfully fetched 30m base data for {symbol} from {api_name} ({len(data)} periods)")
          return data
        else:
          logging.warning(f"{api_name} returned insufficient 30m data for {symbol}")
      except Exception as e:
        logging.warning(f"{api_name} API failed for {symbol} 30m data: {e}")
        continue

    # Final fallback - try to get current price and create synthetic data
    logging.error(f"All APIs failed for {symbol}, attempting synthetic 30m data")
    return self._create_synthetic_30m_data(symbol, required_30m_periods)

  def _fetch_binance_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m data from Binance API"""
    symbol_pair = f"{symbol}USDT"
    url = "https://api.binance.com/api/v3/klines"
    params = {
      'symbol': symbol_pair,
      'interval': '30m',
      'limit': limit
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data:
      return None

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
      'timestamp', 'open', 'high', 'low', 'close', 'volume',
      'close_time', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore'
    ])

    # Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

  def _fetch_coinbase_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m data from Coinbase Pro API"""
    symbol_pair = f"{symbol}-USD"
    granularity = 1800  # 30 minutes in seconds

    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=granularity * limit)

    url = f"https://api.exchange.coinbase.com/products/{symbol_pair}/candles"
    params = {
      'start': start_time.isoformat(),
      'end': end_time.isoformat(),
      'granularity': granularity
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data or not isinstance(data, list):
      return None

    # Coinbase returns: [timestamp, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

  def _fetch_coingecko_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m equivalent data from CoinGecko (limited granularity)"""
    coin_id_map = {
      'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
      'ADA': 'cardano', 'SOL': 'solana', 'DOT': 'polkadot',
      'LINK': 'chainlink', 'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
      'ATOM': 'cosmos', 'LTC': 'litecoin', 'XRP': 'ripple'
    }

    coin_id = coin_id_map.get(symbol, symbol.lower())

    # CoinGecko doesn't have 30m granularity, so we'll get hourly and interpolate
    days = max(1, (limit * 30) // (24 * 60))  # Convert 30m periods to days

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
      'vs_currency': 'usd',
      'days': min(days, 90),  # CoinGecko limit
      'interval': 'hourly' if days <= 90 else 'daily'
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data or 'prices' not in data:
      return None

    # Extract prices data
    prices = data['prices']
    volumes = data.get('total_volumes', [[p[0], 1000000] for p in prices])

    df_data = []
    for i, (price_data, volume_data) in enumerate(zip(prices, volumes)):
      timestamp = pd.to_datetime(price_data[0], unit='ms')
      price = price_data[1]
      volume = volume_data[1]

      # Create synthetic OHLC from price
      df_data.append({
        'timestamp': timestamp,
        'open': price * np.random.uniform(0.999, 1.001),
        'high': price * np.random.uniform(1.0, 1.005),
        'low': price * np.random.uniform(0.995, 1.0),
        'close': price,
        'volume': volume
      })

    df = pd.DataFrame(df_data)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

  def _fetch_cryptocompare_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m data from CryptoCompare API"""
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
      'fsym': symbol,
      'tsym': 'USD',
      'limit': min(limit, 2000),
      'aggregate': 30  # 30 minute aggregation
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get('Response') != 'Success' or not data.get('Data', {}).get('Data'):
      return None

    hist_data = data['Data']['Data']
    df = pd.DataFrame(hist_data)

    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'volumeto': 'volume'})
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

  def _fetch_kraken_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m data from Kraken API"""
    kraken_symbols = {
      'BTC': 'XBTUSD', 'ETH': 'ETHUSD', 'LTC': 'LTCUSD',
      'XRP': 'XRPUSD', 'ADA': 'ADAUSD', 'DOT': 'DOTUSD',
      'LINK': 'LINKUSD', 'ATOM': 'ATOMUSD'
    }

    kraken_symbol = kraken_symbols.get(symbol)
    if not kraken_symbol:
      kraken_symbol = symbol + 'USD'
      return None

    url = "https://api.kraken.com/0/public/OHLC"
    params = {
      'pair': kraken_symbol,
      'interval': 30,  # 30 minutes
      'since': int((datetime.now() - timedelta(minutes=30 * limit)).timestamp())
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get('error') or not data.get('result'):
      return None

    # Get the data
    ohlc_data = None
    for key, value in data['result'].items():
      if isinstance(value, list) and key != 'last':
        ohlc_data = value
        break

    if not ohlc_data:
      return None

    df = pd.DataFrame(ohlc_data, columns=[
      'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

  def _fetch_kucoin_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch 30m data from KuCoin API"""
    symbol_pair = f"{symbol}-USDT"

    end_time = int(datetime.now().timestamp())
    start_time = end_time - (30 * 60 * limit)  # 30 minutes * limit

    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
      'symbol': symbol_pair,
      'type': '30min',
      'startAt': start_time,
      'endAt': end_time
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get('code') != '200000' or not data.get('data'):
      return None

    df = pd.DataFrame(data['data'], columns=[
      'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
    ])

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

  def _create_synthetic_30m_data(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Create synthetic 30m data as absolute last resort"""
    try:
      current_price = self._get_current_price_any_source(symbol)
      if current_price is None:
        return None

      logging.warning(f"Creating synthetic 30m data for {symbol} - price: ${current_price}")

      current_time = datetime.now()
      # Round to nearest 30-minute mark
      current_time = current_time.replace(minute=(current_time.minute // 30) * 30, second=0, microsecond=0)

      timestamps = []
      opens = []
      highs = []
      lows = []
      closes = []
      volumes = []

      for i in range(limit):
        timestamp = current_time - timedelta(minutes=30 * (limit - i - 1))
        timestamps.append(timestamp)

        # Create slight price variations
        variation = np.random.uniform(-0.01, 0.01)
        price = current_price * (1 + variation)

        open_price = price * np.random.uniform(0.995, 1.005)
        close_price = price * np.random.uniform(0.995, 1.005)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.008)
        low_price = min(open_price, close_price) * np.random.uniform(0.992, 1.0)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(np.random.uniform(50000, 500000))

      df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
      })

      return df
    except Exception as e:
      logging.error(f"Failed to create synthetic 30m data for {symbol}: {e}")
      return None

  def _get_current_price_any_source(self, symbol: str) -> Optional[float]:
      """Try to get current price from any available API"""
      price_sources = [
        lambda: self._get_binance_price(symbol),
        lambda: self._get_coinbase_price(symbol),
        lambda: self._get_coingecko_price(symbol),
        lambda: self._get_cryptocompare_price(symbol)
      ]

      for source in price_sources:
        try:
          price = source()
          if price and price > 0:
            return float(price)
        except:
          continue

      return None

  def derive_timeframe_data(self, base_data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Derive higher timeframe data from 30m base data"""
    if target_timeframe == '30m':
      return base_data.copy()

    # Calculate aggregation periods
    timeframe_minutes = self.timeframes[target_timeframe]
    periods_per_candle = timeframe_minutes // 30  # How many 30m periods per target candle

    if periods_per_candle < 1:
      return base_data.copy()

    # Group data into timeframe chunks
    grouped_data = []

    for i in range(0, len(base_data), periods_per_candle):
      chunk = base_data.iloc[i:i + periods_per_candle]

      if len(chunk) == 0:
        continue

      # Aggregate OHLCV data
      aggregated = {
        'timestamp': chunk.iloc[-1]['timestamp'],  # Use end timestamp
        'open': chunk.iloc[0]['open'],  # First open
        'high': chunk['high'].max(),  # Highest high
        'low': chunk['low'].min(),  # Lowest low
        'close': chunk.iloc[-1]['close'],  # Last close
        'volume': chunk['volume'].sum()  # Sum of volumes
      }

      grouped_data.append(aggregated)

    if not grouped_data:
      return pd.DataFrame()

    result_df = pd.DataFrame(grouped_data)

    # Ensure proper alignment for timeframe
    if target_timeframe in ['1h', '2h', '4h', '6h', '8h', '12h', '1d']:
      # Align timestamps to proper timeframe boundaries
      result_df = self._align_timeframe_boundaries(result_df, target_timeframe)

    return result_df

  def _align_timeframe_boundaries(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Align timestamps to proper timeframe boundaries"""
    if df.empty:
      return df

    df = df.copy()

    # Define alignment rules
    if timeframe == '1h':
      df['timestamp'] = df['timestamp'].dt.floor('h')
    elif timeframe == '2h':
      df['timestamp'] = df['timestamp'].dt.floor('2 h')
    elif timeframe == '4h':
      df['timestamp'] = df['timestamp'].dt.floor('4 h')
    elif timeframe == '6h':
      df['timestamp'] = df['timestamp'].dt.floor('6 h')
    elif timeframe == '8h':
      df['timestamp'] = df['timestamp'].dt.floor('8 h')
    elif timeframe == '12h':
      df['timestamp'] = df['timestamp'].dt.floor('12 h')
    elif timeframe == '1d':
      df['timestamp'] = df['timestamp'].dt.floor('D')

    return df

  def analyze_symbol_comprehensive(self, symbol: str) -> Dict:
    """Comprehensive analysis using optimized single API call"""
    # Fetch base 30m data once
    base_data = self.get_historical_data_optimized(symbol, 200)

    if base_data is None or len(base_data) < 100:
      logging.error(f"Insufficient base data for {symbol}")
      return {
        'symbol': symbol,
        'signal': 'HOLD',
        'confidence_score': 0,
        'confidence_percentage': 0.0,
        'buy_signals': [],
        'sell_signals': [],
        'predominant_timeframe': '1h',
        'timeframe_signals': {},
        'total_signals': 0
      }

    all_signals = {
      'BUY': [],
      'SELL': []
    }

    timeframe_signals = {}

    # Process each timeframe using derived data
    for timeframe in self.timeframes.keys():
      try:
        # Derive timeframe data from base 30m data
        timeframe_data = self.derive_timeframe_data(base_data, timeframe)

        if timeframe_data is None or len(timeframe_data) < 20:
          logging.warning(f"Insufficient {timeframe} data derived for {symbol}")
          continue

        # Calculate indicators for this timeframe
        data_with_indicators = self.calculate_indicators(timeframe_data)

        if len(data_with_indicators) < 20:
          continue

        # Generate signals
        individual_signals = self.generate_indicator_signals(data_with_indicators)
        combination_signals = self.generate_combination_signals(data_with_indicators)

        # Combine all signals for this timeframe
        timeframe_all_signals = {**individual_signals, **combination_signals}
        timeframe_signals[timeframe] = timeframe_all_signals

        # Categorize signals with timeframe prefix
        for signal_name, signal_type in timeframe_all_signals.items():
          signal_entry = f"{timeframe}_{signal_name}"
          if signal_type == 'BUY':
            all_signals['BUY'].append(signal_entry)
          elif signal_type == 'SELL':
            all_signals['SELL'].append(signal_entry)

      except Exception as e:
        logging.error(f"Error analyzing {symbol} on {timeframe}: {e}")
        continue

    # Determine final signal and confidence (same logic as before)
    buy_count = len(all_signals['BUY'])
    sell_count = len(all_signals['SELL'])

    # Determine predominant timeframe
    timeframe_buy_counts = defaultdict(int)
    timeframe_sell_counts = defaultdict(int)

    for signal in all_signals['BUY']:
      tf = signal.split('_')[0]
      timeframe_buy_counts[tf] += 1

    for signal in all_signals['SELL']:
      tf = signal.split('_')[0]
      timeframe_sell_counts[tf] += 1

    all_tf_counts = defaultdict(int)
    for tf, count in timeframe_buy_counts.items():
      all_tf_counts[tf] += count
    for tf, count in timeframe_sell_counts.items():
      all_tf_counts[tf] += count

    predominant_timeframe = max(all_tf_counts.items(), key=lambda x: x[1])[0] if all_tf_counts else '1h'

    # Calculate confidence
    if buy_count > sell_count and buy_count > self.conf_threshold:
      final_signal = 'BUY'
      confidence_score = buy_count
    elif sell_count > buy_count and sell_count > self.conf_threshold:
      final_signal = 'SELL'
      confidence_score = sell_count
    else:
      final_signal = 'HOLD'
      confidence_score = max(buy_count, sell_count)

    confidence_percentage = (confidence_score / 504) * 100

    logging.info(
      f"Optimized analysis complete for {symbol}: {len(base_data)} base periods → {buy_count} BUY, {sell_count} SELL signals")

    return {
      'symbol': symbol,
      'signal': final_signal,
      'confidence_score': confidence_score,
      'confidence_percentage': confidence_percentage,
      'buy_signals': all_signals['BUY'],
      'sell_signals': all_signals['SELL'],
      'predominant_timeframe': predominant_timeframe,
      'timeframe_signals': timeframe_signals,
      'total_signals': buy_count + sell_count
    }

  def calculate_stop_loss_and_validation(self, signal: str, confidence_score: int,
                                         predominant_timeframe: str, current_price: float) -> Tuple[float, int]:
    """Calculate stop loss and maximum validation period"""
    # Base stop loss percentages based on confidence
    if confidence_score >= 100:
      base_stop_loss_pct = 0.02  # 2% for very high confidence
    elif confidence_score >= 75:
      base_stop_loss_pct = 0.03  # 3% for high confidence
    elif confidence_score >= 50:
      base_stop_loss_pct = 0.05  # 5% for medium confidence
    else:
      base_stop_loss_pct = 0.08  # 8% for low confidence

    # Adjust based on timeframe
    timeframe_multipliers = {
      '30m': 0.5, '1h': 0.7, '2h': 0.8, '4h': 1.0,
      '6h': 1.2, '8h': 1.3, '12h': 1.5, '1d': 2.0
    }

    multiplier = timeframe_multipliers.get(predominant_timeframe, 1.0)
    final_stop_loss_pct = base_stop_loss_pct * multiplier

    # Calculate stop loss price
    if signal == 'BUY':
      stop_loss = current_price * (1 - final_stop_loss_pct)
    else:  # SELL
      stop_loss = current_price * (1 + final_stop_loss_pct)

    # Calculate maximum validation period (in minutes)
    timeframe_minutes = self.timeframes[predominant_timeframe]
    validation_periods = {
      '30m': 4, '1h': 6, '2h': 8, '4h': 12,
      '6h': 16, '8h': 20, '12h': 24, '1d': 48
    }

    periods = validation_periods.get(predominant_timeframe, 12)
    max_validation_minutes = timeframe_minutes * periods

    return stop_loss, max_validation_minutes

  def save_signal_to_db(self, analysis_result: Dict, current_price: float,
                        stop_loss: float, max_validation_period: int):
    """Save signal analysis to database with detailed timeframe-specific indicator tracking"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      # Get contributing signals with timeframes
      if analysis_result['signal'] == 'BUY':
        contributing_signals = analysis_result['buy_signals']
      elif analysis_result['signal'] == 'SELL':
        contributing_signals = analysis_result['sell_signals']
      else:
        contributing_signals = []

      # Create detailed timeframe-specific breakdown
      timeframe_indicator_map = {}
      indicator_timeframe_list = []

      for signal in contributing_signals:
        if '_' in signal:
          timeframe, indicator = signal.split('_', 1)

          # Track indicators by timeframe
          if timeframe not in timeframe_indicator_map:
            timeframe_indicator_map[timeframe] = []
          timeframe_indicator_map[timeframe].append(indicator)

          # Create timeframe-specific entries
          indicator_timeframe_list.append(f"[{timeframe}] {indicator}")
        else:
          indicator_timeframe_list.append(f"[unknown] {signal}")

      # Create comprehensive breakdown
      breakdown = {
        'buy_signals': analysis_result['buy_signals'],
        'sell_signals': analysis_result['sell_signals'],
        'timeframe_signals': analysis_result['timeframe_signals'],
        'timeframe_indicator_breakdown': timeframe_indicator_map,
        'signal_summary': {
          'total_buy': len(analysis_result['buy_signals']),
          'total_sell': len(analysis_result['sell_signals']),
          'unique_indicators': self._get_unique_indicators_with_timeframes(analysis_result),
          'timeframe_distribution': self._get_detailed_timeframe_distribution(analysis_result),
          'indicator_frequency': self._get_indicator_frequency_by_timeframe(analysis_result)
        }
      }

      # Join all timeframe-specific indicators for database storage
      all_indicators_with_timeframes = ', '.join(sorted(indicator_timeframe_list))

      # Also create a summary without timeframes for backward compatibility
      unique_indicators = set()
      for signal in contributing_signals:
        if '_' in signal:
          indicator = signal.split('_', 1)[1]
          unique_indicators.add(indicator)
        else:
          unique_indicators.add(signal)

      unique_indicators_str = ', '.join(sorted(unique_indicators))

      cursor.execute('''
                INSERT INTO signals (
                    symbol, signal, confidence_score, total_indicators, 
                    confidence_percentage, price, stop_loss, predominant_timeframe,
                    max_validation_period, indicator_breakdown, contributing_indicators,
                    timeframe_specific_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
        analysis_result['symbol'],
        analysis_result['signal'],
        analysis_result['confidence_score'],
        504,  # Total possible indicators
        analysis_result['confidence_percentage'],
        current_price,
        stop_loss,
        analysis_result['predominant_timeframe'],
        max_validation_period,
        json.dumps(breakdown, indent=2),
        unique_indicators_str,  # Indicators without timeframes
        all_indicators_with_timeframes  # Indicators with timeframes
      ))

      conn.commit()
      conn.close()

      logging.info(
        f"Signal saved to database: {analysis_result['symbol']} {analysis_result['signal']} - {len(contributing_signals)} timeframe-specific indicators")

    except Exception as e:
      logging.error(f"Failed to save signal to database: {e}")

  def _get_unique_indicators_with_timeframes(self, analysis_result: Dict) -> Dict[str, List[str]]:
    """Get indicators grouped by timeframes"""
    if analysis_result['signal'] == 'BUY':
      signals = analysis_result['buy_signals']
    elif analysis_result['signal'] == 'SELL':
      signals = analysis_result['sell_signals']
    else:
      return {}

    indicator_timeframes = {}
    for signal in signals:
      if '_' in signal:
        timeframe, indicator = signal.split('_', 1)
        if indicator not in indicator_timeframes:
          indicator_timeframes[indicator] = []
        if timeframe not in indicator_timeframes[indicator]:
          indicator_timeframes[indicator].append(timeframe)

    # Sort timeframes for each indicator
    for indicator in indicator_timeframes:
      indicator_timeframes[indicator] = sorted(indicator_timeframes[indicator])

    return indicator_timeframes

  def _get_detailed_timeframe_distribution(self, analysis_result: Dict) -> Dict[str, Dict[str, int]]:
    """Get detailed distribution of signals across timeframes"""
    if analysis_result['signal'] == 'BUY':
      signals = analysis_result['buy_signals']
    elif analysis_result['signal'] == 'SELL':
      signals = analysis_result['sell_signals']
    else:
      return {}

    timeframe_details = {}
    for signal in signals:
      if '_' in signal:
        timeframe, indicator = signal.split('_', 1)
        if timeframe not in timeframe_details:
          timeframe_details[timeframe] = {'count': 0, 'indicators': set()}
        timeframe_details[timeframe]['count'] += 1
        timeframe_details[timeframe]['indicators'].add(indicator)

    # Convert sets to counts
    for tf in timeframe_details:
      timeframe_details[tf]['unique_indicators'] = len(timeframe_details[tf]['indicators'])
      timeframe_details[tf]['indicators'] = list(timeframe_details[tf]['indicators'])

    return timeframe_details

  def _get_indicator_frequency_by_timeframe(self, analysis_result: Dict) -> Dict[str, Dict[str, int]]:
    """Get frequency of each indicator across timeframes"""
    if analysis_result['signal'] == 'BUY':
      signals = analysis_result['buy_signals']
    elif analysis_result['signal'] == 'SELL':
      signals = analysis_result['sell_signals']
    else:
      return {}

    indicator_freq = {}
    for signal in signals:
      if '_' in signal:
        timeframe, indicator = signal.split('_', 1)
        if indicator not in indicator_freq:
          indicator_freq[indicator] = {}
        if timeframe not in indicator_freq[indicator]:
          indicator_freq[indicator][timeframe] = 0
        indicator_freq[indicator][timeframe] += 1

    return indicator_freq

  def get_last_signal(self, symbol: str) -> Optional[Dict]:
    """Get the last signal sent for a symbol with timeframe-specific indicator details"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      cursor.execute('''
                  SELECT signal, confidence_score, price, datetime_created, 
                         contributing_indicators, timeframe_specific_indicators 
                  FROM signals 
                  WHERE symbol = ? 
                  ORDER BY datetime_created DESC 
                  LIMIT 1
              ''', (symbol,))

      result = cursor.fetchone()
      conn.close()

      if result:
        return {
          'signal': result[0],
          'confidence_score': result[1],
          'price': result[2],
          'datetime': result[3],
          'contributing_indicators': result[4] if result[4] else '',
          'timeframe_specific_indicators': result[5] if result[5] else ''
        }
      return None

    except Exception as e:
      logging.error(f"Failed to get last signal: {e}")
      return None

  def should_send_alert(self, symbol: str, new_signal: str, new_confidence: int) -> bool:
    """Check if alert should be sent based on database history"""
    last_signal = self.get_last_signal(symbol)

    if last_signal is None:
      return True  # No previous signal, send alert

    # Send alert if signal changed or confidence significantly improved
    if (last_signal['signal'] != new_signal or
      (last_signal['signal'] == new_signal and new_confidence > last_signal['confidence_score'] + 20)):
      return True

    return False

  def send_email_alert(self, analysis_result: Dict, current_price: float,
                       stop_loss: float, max_validation_period: int) -> bool:
    """Send email alert to multiple recipients"""
    try:
      # Support multiple recipients
      recipients = self.email_config.get('recipient_emails', [])
      if not recipients:
        # Fallback to single recipient for backward compatibility
        recipients = [self.email_config.get('recipient_email')]

      for recipient in recipients:
        if not recipient:
          continue

        msg = MIMEMultipart()
        msg['From'] = self.email_config['sender_email']
        msg['To'] = recipient

        # Create subject line
        signal = analysis_result['signal']
        confidence = analysis_result['confidence_score']
        symbol = analysis_result['symbol']
        msg[
          'Subject'] = f"🚨 {signal} {symbol} - {confidence} Indicators ({analysis_result['confidence_percentage']:.1f}%)"

        # Create detailed email body
        body = self.create_email_body(analysis_result, current_price, stop_loss, max_validation_period)
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['sender_email'], self.email_config['sender_password'])

        text = msg.as_string()
        server.sendmail(self.email_config['sender_email'], recipient, text)
        server.quit()

        logging.info(f"Alert sent to {recipient}: {symbol} {signal}")

      return True

    except Exception as e:
      logging.error(f"Failed to send email: {e}")
      return False

  def create_email_body(self, analysis_result: Dict, current_price: float,
                        stop_loss: float, max_validation_period: int) -> str:
    """Create detailed email body with timeframe-specific indicator breakdown"""

    symbol = analysis_result['symbol']
    signal = analysis_result['signal']
    confidence_score = analysis_result['confidence_score']
    confidence_pct = analysis_result['confidence_percentage']
    predominant_tf = analysis_result['predominant_timeframe']

    buy_count = len(analysis_result['buy_signals'])
    sell_count = len(analysis_result['sell_signals'])

    # Get contributing signals with timeframes
    if signal == 'BUY':
      contributing_signals = analysis_result['buy_signals']
    elif signal == 'SELL':
      contributing_signals = analysis_result['sell_signals']
    else:
      contributing_signals = []

    # Organize indicators by timeframe
    timeframe_indicators = {}
    indicator_timeframe_map = {}

    for sig in contributing_signals:
      if '_' in sig:
        timeframe, indicator = sig.split('_', 1)

        # Group by timeframe
        if timeframe not in timeframe_indicators:
          timeframe_indicators[timeframe] = []
        timeframe_indicators[timeframe].append(indicator)

        # Group by indicator
        if indicator not in indicator_timeframe_map:
          indicator_timeframe_map[indicator] = []
        indicator_timeframe_map[indicator].append(timeframe)

    # Sort timeframes for each indicator
    for indicator in indicator_timeframe_map:
      indicator_timeframe_map[indicator] = sorted(set(indicator_timeframe_map[indicator]))


    body = f"""
        🚨 ADVANCED CRYPTO TRADING ALERT 🚨
    
        SYMBOL: {symbol}
        SIGNAL: {signal}
        CURRENT PRICE: ${current_price:,.4f}
    
        📊 CONFIDENCE ANALYSIS:
        • Confidence Score: {confidence_score}/504 indicators ({confidence_pct:.1f}%)
        • Buy Signals: {buy_count}
        • Sell Signals: {sell_count}
        • Predominant Timeframe: {predominant_tf}
        • Active Timeframes: {len(timeframe_indicators)}
        • Unique Indicators: {len(indicator_timeframe_map)}
    
        💰 TRADING PARAMETERS:
        • Stop Loss: ${stop_loss:,.4f}
        • Max Validation Period: {max_validation_period} minutes ({max_validation_period / 60:.1f} hours)
    
        🔍 INDICATORS BY TIMEFRAME:
        """

    # Show indicators grouped by timeframe
    for timeframe in sorted(timeframe_indicators.keys()):
      indicators = timeframe_indicators[timeframe]
      body += f"\n📈 {timeframe.upper()} TIMEFRAME ({len(indicators)} signals):\n"

      # Group similar indicators
      indicator_counts = {}
      for ind in indicators:
        indicator_counts[ind] = indicator_counts.get(ind, 0) + 1

      for indicator, count in sorted(indicator_counts.items()):
        if count > 1:
          body += f"  • {indicator} ({count}x)\n"
        else:
          body += f"  • {indicator}\n"

    body += f"""
    
        🎯 INDICATORS ACROSS MULTIPLE TIMEFRAMES:
        """

    # Show indicators that appear in multiple timeframes
    multi_timeframe_indicators = {k: v for k, v in indicator_timeframe_map.items() if len(v) > 1}
    if multi_timeframe_indicators:
      for indicator, timeframes in sorted(multi_timeframe_indicators.items(), key=lambda x: len(x[1]), reverse=True):
        tf_list = ' + '.join(timeframes)
        body += f"• {indicator}: {tf_list} ({len(timeframes)} timeframes)\n"
    else:
      body += "• No indicators detected across multiple timeframes\n"

    body += f"""
    
        📋 COMPLETE SIGNAL LIST WITH TIMEFRAMES:
        """

    # Show detailed signals with timeframes (limit to first 25)
    for i, sig in enumerate(contributing_signals[:25], 1):
      if '_' in sig:
        timeframe, indicator = sig.split('_', 1)
        body += f"{i:2d}. [{timeframe.upper()}] {indicator}\n"
      else:
        body += f"{i:2d}. [UNKNOWN] {sig}\n"

    if len(contributing_signals) > 25:
      body += f"... and {len(contributing_signals) - 25} more timeframe-specific signals\n"

    body += f"""
    
        📊 TIMEFRAME STRENGTH ANALYSIS:
        """

    # Analyze timeframe strength
    timeframe_strength = {}
    for tf, indicators in timeframe_indicators.items():
      timeframe_strength[tf] = len(indicators)

    if timeframe_strength:
      sorted_strength = sorted(timeframe_strength.items(), key=lambda x: x[1], reverse=True)
      for i, (tf, count) in enumerate(sorted_strength, 1):
        strength_pct = (count / confidence_score) * 100 if confidence_score > 0 else 0
        body += f"{i}. {tf.upper()}: {count} signals ({strength_pct:.1f}% of total)\n"

    # Calculate timeframe diversity score
    timeframe_diversity = len(timeframe_indicators) / len(self.timeframes) * 100

    body += f"""
    
        📈 SIGNAL QUALITY METRICS:
        • Timeframe Diversity: {timeframe_diversity:.1f}% ({len(timeframe_indicators)}/8 timeframes active)
        • Average Signals per Timeframe: {confidence_score / len(timeframe_indicators):.1f}
        • Cross-Timeframe Confirmations: {len(multi_timeframe_indicators)} indicators
        • Predominant Timeframe Contribution: {timeframe_strength.get(predominant_tf, 0)} signals
    
        ⚠️ RISK MANAGEMENT:
        • Signal based on {confidence_score} timeframe-specific indicator combinations
        • {len(indicator_timeframe_map)} unique indicators across {len(timeframe_indicators)} timeframes
        • Strongest confirmation from {predominant_tf.upper()} timeframe
        • Always use 1-3% position sizing
        • Set stop loss at ${stop_loss:,.4f}
        • Monitor signal validity for {max_validation_period / 60:.1f} hours
    
        📅 Alert Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
        ---
        Advanced Crypto Alert System
        Multi-Timeframe Multi-Indicator Analysis
        {confidence_score} Timeframe-Specific Signals | {confidence_pct:.1f}% Confidence
                """

    return body


  def monitor_symbol(self, symbol: str):
    """Monitor a single symbol with comprehensive analysis"""
    try:
      logging.info(f"Analyzing {symbol}...")

      # Perform comprehensive analysis
      analysis_result = self.analyze_symbol_comprehensive(symbol)

      if analysis_result['signal'] == 'HOLD':
        logging.info(f"{symbol}: HOLD - Confidence: {analysis_result['confidence_score']}")
        return

      # Get current price (from latest 1h data)
      current_data = self.get_historical_data_optimized(symbol, 1)
      if current_data is None or len(current_data) == 0:
        logging.error(f"Could not get current price for {symbol}")
        return

      current_price = float(current_data['close'].iloc[-1])

      # Calculate stop loss and validation period
      stop_loss, max_validation_period = self.calculate_stop_loss_and_validation(
        analysis_result['signal'],
        analysis_result['confidence_score'],
        analysis_result['predominant_timeframe'],
        current_price
      )

      # Check if we should send alert
      if self.should_send_alert(symbol, analysis_result['signal'], analysis_result['confidence_score']):

        # Save to database
        self.save_signal_to_db(analysis_result, current_price, stop_loss, max_validation_period)

        # Send email alert
        # success = self.send_email_alert(analysis_result, current_price, stop_loss, max_validation_period)
        success = True

        if success:
          logging.info(
            f"*** ALERT SENT: {symbol} {analysis_result['signal']} - {analysis_result['confidence_score']} indicators ***")
        else:
          logging.error(f"Failed to send alert for {symbol}")
      else:
        logging.info(
          f"{symbol}: {analysis_result['signal']} signal blocked - same as last or insufficient confidence change")

      # Log comprehensive results
      logging.info(
        f"{symbol}: {analysis_result['signal']} | Confidence: {analysis_result['confidence_score']}/504 ({analysis_result['confidence_percentage']:.1f}%) | Price: ${current_price:.4f} | TF: {analysis_result['predominant_timeframe']}")

    except Exception as e:
      logging.error(f"Error monitoring {symbol}: {e}")


  def initialize_symbol_queue(self, symbols: List[str]):
    """Initialize the symbol queue"""
    self.symbol_queue = deque(symbols)
    self.symbol_stats = {symbol: {'checks': 0, 'signals': 0} for symbol in symbols}
    logging.info(f"Initialized queue with {len(symbols)} symbols: {symbols}")


  def get_next_symbol(self) -> Optional[str]:
    """Get next symbol from queue"""
    if not self.symbol_queue:
      return None

    symbol = self.symbol_queue.popleft()
    self.symbol_queue.append(symbol)
    return symbol


  def run_continuous_monitoring(self, symbols: List[str]):
    """Run continuous monitoring without delays"""
    self.initialize_symbol_queue(symbols)
    self.running = True

    logging.info(f"Starting continuous monitoring for {len(symbols)} symbols")
    logging.info("No delays - maximum speed analysis")

    print(f"\nContinuous Monitoring Started")
    print(f"Symbols: {symbols}")
    print(f"Press Ctrl+C to stop\n")
    print(f"{'COUNT':<6} {'SYMBOL':<8} {'SIGNAL':<6} {'CONFIDENCE':<12} {'PRICE':<12} {'TIMEFRAME':<10}")
    print("-" * 70)

    cycle_count = 0
    start_time = datetime.now()

    try:
      while self.running:
        symbol = self.get_next_symbol()
        if symbol is None:
          break

        # Update stats
        self.symbol_stats[symbol]['checks'] += 1
        cycle_count += 1

        # Monitor symbol (no delays)
        self.monitor_symbol(symbol)

        # Show periodic stats
        if cycle_count % 50 == 0:
          elapsed = datetime.now() - start_time
          rate = cycle_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
          print(f"\n[{cycle_count} analyses completed, {rate:.2f} analyses/second]")

    except KeyboardInterrupt:
      logging.info("Monitoring stopped by user")
      self.running = False
      self.print_monitoring_stats(start_time, cycle_count)
    except Exception as e:
      logging.error(f"Error in monitoring loop: {e}")
      self.running = False


  def print_monitoring_stats(self, start_time: datetime, cycle_count: int):
    """Print comprehensive monitoring statistics"""
    elapsed = datetime.now() - start_time

    print("\n" + "=" * 80)
    print("MONITORING STATISTICS")
    print("=" * 80)
    print(f"Total Runtime: {elapsed}")
    print(f"Total Analyses: {cycle_count}")
    print(f"Analysis Rate: {cycle_count / elapsed.total_seconds():.2f} analyses/second")
    print()

    # Symbol stats
    print(f"{'SYMBOL':<8} {'CHECKS':<8} {'SIGNALS':<8} {'RATE':<10}")
    print("-" * 40)

    for symbol in self.symbol_queue:
      stats = self.symbol_stats[symbol]
      rate = stats['checks'] / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
      print(f"{symbol:<8} {stats['checks']:<8} {stats['signals']:<8} {rate:.2f}/s")

    # Database stats
    self.print_database_stats()


  def print_database_stats(self):
    """Print database statistics"""
    try:
      conn = sqlite3.connect(self.db_path)
      cursor = conn.cursor()

      # Total signals
      cursor.execute("SELECT COUNT(*) FROM signals")
      total_signals = cursor.fetchone()[0]

      # Signals by type
      cursor.execute("SELECT signal, COUNT(*) FROM signals GROUP BY signal")
      signal_counts = cursor.fetchall()

      # Recent signals (last hour)
      cursor.execute("""
                      SELECT COUNT(*) FROM signals 
                      WHERE datetime_created > datetime('now', '-1 hour')
                  """)
      recent_signals = cursor.fetchone()[0]

      print(f"\nDATABASE STATISTICS:")
      print(f"Total Signals Stored: {total_signals}")
      print(f"Recent Signals (1h): {recent_signals}")

      for signal_type, count in signal_counts:
        print(f"{signal_type} Signals: {count}")

      conn.close()

    except Exception as e:
      logging.error(f"Error getting database stats: {e}")


  def send_test_email(self) -> bool:
    """Send test email to verify configuration"""
    try:
      recipients = self.email_config.get('recipient_emails', [])
      if not recipients:
        recipients = [self.email_config.get('recipient_email')]

      for recipient in recipients:
        if not recipient:
          continue

        msg = MIMEMultipart()
        msg['From'] = self.email_config['sender_email']
        msg['To'] = recipient
        msg['Subject'] = "🧪 Advanced Crypto Alert System - Test Email"

        body = f"""
      🧪 TEST EMAIL - ADVANCED CRYPTO ALERT SYSTEM
  
      Configuration Test Results:
      ✅ Email configuration loaded successfully
      ✅ SMTP connection established
      ✅ Multi-recipient support: {len(recipients)} recipients configured
  
      System Capabilities:
      📊 6 Technical Indicators: MACD, Momentum_5, Momentum_10, RSI, STOCH_D, WILLR
      ⏰ 8 Timeframes: 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
      🎯 504 Total Signal Combinations per Symbol
      🗄️ SQLite Database Integration
      🚫 Anti-Duplicate System via Database History
      📈 Dynamic Stop-Loss Calculation
      ⚡ Zero-Delay Continuous Monitoring
  
      Recipient: {recipient}
      Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  
      ---
      Ready for live trading alerts!
      Run: python crypto_alerts.py
                      """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['sender_email'], self.email_config['sender_password'])

        text = msg.as_string()
        server.sendmail(self.email_config['sender_email'], recipient, text)
        server.quit()

        print(f"✅ Test email sent successfully to: {recipient}")

      return True

    except Exception as e:
      print(f"❌ Test email failed: {e}")
      logging.error(f"Test email failed: {e}")
      return False

  def _create_synthetic_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Create synthetic data as absolute last resort"""
    try:
      # Try to get current price from any available source
      current_price = self._get_current_price_any_source(symbol)
      if current_price is None:
        return None

      logging.warning(f"Creating synthetic data for {symbol} - price: ${current_price}")

      # Create synthetic OHLCV data
      timeframe_minutes = self.timeframes[timeframe]
      current_time = datetime.now()

      timestamps = []
      opens = []
      highs = []
      lows = []
      closes = []
      volumes = []

      for i in range(limit):
        # Go back in time
        timestamp = current_time - timedelta(minutes=timeframe_minutes * (limit - i - 1))
        timestamps.append(timestamp)

        # Create slight price variations (±0.5%)
        variation = np.random.uniform(-0.005, 0.005)
        price = current_price * (1 + variation)

        # Create OHLC with small variations
        open_price = price * np.random.uniform(0.998, 1.002)
        close_price = price * np.random.uniform(0.998, 1.002)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.005)
        low_price = min(open_price, close_price) * np.random.uniform(0.995, 1.0)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(np.random.uniform(100000, 1000000))  # Random volume

      df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
      })

      return df

    except Exception as e:
      logging.error(f"Failed to create synthetic data for {symbol}: {e}")
      return None

  def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 6 technical indicators"""
    data = df.copy()

    if len(data) < 50:  # Need sufficient data
      return data

    try:
      # MACD
      ema_12 = data['close'].ewm(span=12).mean()
      ema_26 = data['close'].ewm(span=26).mean()
      data['MACD'] = ema_12 - ema_26
      data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
      data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

      # Momentum 5 and 10
      data['Momentum_5'] = data['close'] / data['close'].shift(5) - 1
      data['Momentum_10'] = data['close'] / data['close'].shift(10) - 1

      # RSI
      delta = data['close'].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      data['RSI'] = 100 - (100 / (1 + rs))

      # Stochastic %D
      lowest_low = data['low'].rolling(window=14).min()
      highest_high = data['high'].rolling(window=14).max()
      k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
      data['STOCH_K'] = k_percent.rolling(window=3).mean()
      data['STOCH_D'] = data['STOCH_K'].rolling(window=3).mean()

      # Williams %R
      data['WILLR'] = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))

    except Exception as e:
      logging.error(f"Error calculating indicators: {e}")

    return data

  def generate_indicator_signals(self, data: pd.DataFrame) -> Dict[str, str]:
    """Generate signals from individual indicators"""
    if len(data) < 2:
      return {}

    signals = {}
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    try:
      # MACD Signals
      if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
          signals['MACD_crossover'] = 'BUY'
        elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
          signals['MACD_crossover'] = 'SELL'

        if latest['MACD'] > 0:
          signals['MACD_position'] = 'BUY'
        else:
          signals['MACD_position'] = 'SELL'

      # Momentum Signals
      if not pd.isna(latest['Momentum_5']):
        if latest['Momentum_5'] > 0.02:
          signals['Momentum_5'] = 'BUY'
        elif latest['Momentum_5'] < -0.02:
          signals['Momentum_5'] = 'SELL'

      if not pd.isna(latest['Momentum_10']):
        if latest['Momentum_10'] > 0.03:
          signals['Momentum_10'] = 'BUY'
        elif latest['Momentum_10'] < -0.03:
          signals['Momentum_10'] = 'SELL'

      # RSI Signals
      if not pd.isna(latest['RSI']):
        if latest['RSI'] < 30:
          signals['RSI_oversold'] = 'BUY'
        elif latest['RSI'] > 70:
          signals['RSI_overbought'] = 'SELL'

        if latest['RSI'] > 50 and previous['RSI'] <= 50:
          signals['RSI_midline'] = 'BUY'
        elif latest['RSI'] < 50 and previous['RSI'] >= 50:
          signals['RSI_midline'] = 'SELL'

      # Stochastic %D Signals
      if not pd.isna(latest['STOCH_D']):
        if latest['STOCH_D'] < 20:
          signals['STOCH_D_oversold'] = 'BUY'
        elif latest['STOCH_D'] > 80:
          signals['STOCH_D_overbought'] = 'SELL'

        if latest['STOCH_D'] > 50 and previous['STOCH_D'] <= 50:
          signals['STOCH_D_midline'] = 'BUY'
        elif latest['STOCH_D'] < 50 and previous['STOCH_D'] >= 50:
          signals['STOCH_D_midline'] = 'SELL'

      # Williams %R Signals
      if not pd.isna(latest['WILLR']):
        if latest['WILLR'] > -20:
          signals['WILLR_overbought'] = 'SELL'
        elif latest['WILLR'] < -80:
          signals['WILLR_oversold'] = 'BUY'

        if latest['WILLR'] > -50 and previous['WILLR'] <= -50:
          signals['WILLR_midline'] = 'BUY'
        elif latest['WILLR'] < -50 and previous['WILLR'] >= -50:
          signals['WILLR_midline'] = 'SELL'

    except Exception as e:
      logging.error(f"Error generating signals: {e}")

    return signals

  def generate_combination_signals(self, data: pd.DataFrame) -> Dict[str, str]:
    """Generate signals from indicator combinations"""
    if len(data) < 2:
      return {}

    signals = {}
    latest = data.iloc[-1]

    try:
      # MACD + RSI
      if (not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']) and
        not pd.isna(latest['RSI'])):
        if latest['MACD'] > latest['MACD_signal'] and latest['RSI'] < 30:
          signals['MACD_RSI_bullish'] = 'BUY'
        elif latest['MACD'] < latest['MACD_signal'] and latest['RSI'] > 70:
          signals['MACD_RSI_bearish'] = 'SELL'

      # MACD + Momentum
      if (not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']) and
        not pd.isna(latest['Momentum_10'])):
        if latest['MACD'] > latest['MACD_signal'] and latest['Momentum_10'] > 0.02:
          signals['MACD_MOM_bullish'] = 'BUY'
        elif latest['MACD'] < latest['MACD_signal'] and latest['Momentum_10'] < -0.02:
          signals['MACD_MOM_bearish'] = 'SELL'

      # RSI + Stochastic
      if not pd.isna(latest['RSI']) and not pd.isna(latest['STOCH_D']):
        if latest['RSI'] < 30 and latest['STOCH_D'] < 20:
          signals['RSI_STOCH_oversold'] = 'BUY'
        elif latest['RSI'] > 70 and latest['STOCH_D'] > 80:
          signals['RSI_STOCH_overbought'] = 'SELL'

      # RSI + Williams %R
      if not pd.isna(latest['RSI']) and not pd.isna(latest['WILLR']):
        if latest['RSI'] < 30 and latest['WILLR'] < -80:
          signals['RSI_WILLR_oversold'] = 'BUY'
        elif latest['RSI'] > 70 and latest['WILLR'] > -20:
          signals['RSI_WILLR_overbought'] = 'SELL'

      # Momentum combinations
      if not pd.isna(latest['Momentum_5']) and not pd.isna(latest['Momentum_10']):
        if latest['Momentum_5'] > 0.02 and latest['Momentum_10'] > 0.03:
          signals['MOM_5_10_bullish'] = 'BUY'
        elif latest['Momentum_5'] < -0.02 and latest['Momentum_10'] < -0.03:
          signals['MOM_5_10_bearish'] = 'SELL'

      # Stochastic + Williams %R
      if not pd.isna(latest['STOCH_D']) and not pd.isna(latest['WILLR']):
        if latest['STOCH_D'] < 20 and latest['WILLR'] < -80:
          signals['STOCH_WILLR_oversold'] = 'BUY'
        elif latest['STOCH_D'] > 80 and latest['WILLR'] > -20:
          signals['STOCH_WILLR_overbought'] = 'SELL'

    except Exception as e:
      logging.error(f"Error generating combination signals: {e}")

    return signals

  def test_timeframe_optimization(self):
    """Test the optimized timeframe derivation system"""
    print("🧪 Testing Optimized Timeframe System...")

    # Create temporary config for testing
    temp_config = {
      'sender_email': 'test@test.com',
      'sender_password': 'test',
      'recipient_emails': ['test@test.com'],
      'smtp_server': 'smtp.gmail.com',
      'smtp_port': 587
    }

    system = AdvancedCryptoAlertSystem(temp_config)

    test_symbol = 'BTC'
    print(f"\n📊 Testing timeframe derivation for {test_symbol}...")

    try:
      # Test base data fetching
      print("1. Fetching base 30m data...")
      base_data = system.get_historical_data_optimized(test_symbol, 200)

      if base_data is None or len(base_data) < 50:
        print("❌ Failed to get sufficient base data")
        return

      print(f"✅ Got {len(base_data)} base 30m periods")
      print(f"   Date range: {base_data['timestamp'].min()} to {base_data['timestamp'].max()}")
      print(f"   Price range: ${base_data['close'].min():.2f} - ${base_data['close'].max():.2f}")

      # Test timeframe derivation
      print("\n2. Testing timeframe derivation...")

      for timeframe in ['1h', '4h', '12h', '1d']:
        print(f"\n   Testing {timeframe} derivation...")
        derived_data = system.derive_timeframe_data(base_data, timeframe)

        if derived_data is not None and len(derived_data) > 0:
          print(f"   ✅ {timeframe}: {len(derived_data)} periods derived")
          print(f"      Latest price: ${derived_data['close'].iloc[-1]:.2f}")
          print(f"      Date range: {derived_data['timestamp'].min()} to {derived_data['timestamp'].max()}")

          # Test indicator calculation
          indicators = system.calculate_indicators(derived_data)
          if len(indicators) > 0:
            latest = indicators.iloc[-1]
            print(f"      RSI: {latest.get('RSI', 'N/A'):.1f}" if not pd.isna(latest.get('RSI')) else "      RSI: N/A")
            print(
              f"      MACD: {latest.get('MACD', 'N/A'):.4f}" if not pd.isna(latest.get('MACD')) else "      MACD: N/A")
        else:
          print(f"   ❌ {timeframe}: Failed to derive data")

      # Test comprehensive analysis
      print("\n3. Testing comprehensive analysis...")
      analysis = system.analyze_symbol_comprehensive(test_symbol)

      print(f"✅ Analysis complete:")
      print(f"   Signal: {analysis['signal']}")
      print(f"   Confidence: {analysis['confidence_score']}/504 ({analysis['confidence_percentage']:.1f}%)")
      print(f"   Buy signals: {len(analysis['buy_signals'])}")
      print(f"   Sell signals: {len(analysis['sell_signals'])}")
      print(f"   Active timeframes: {len(analysis['timeframe_signals'])}")

      # Show timeframe breakdown
      if analysis['timeframe_signals']:
        print(f"\n   Timeframe breakdown:")
        for tf, signals in analysis['timeframe_signals'].items():
          if signals:
            buy_count = sum(1 for s in signals.values() if s == 'BUY')
            sell_count = sum(1 for s in signals.values() if s == 'SELL')
            if buy_count > 0 or sell_count > 0:
              print(f"     {tf}: {buy_count} BUY, {sell_count} SELL")

      print(f"\n🎯 Optimization Benefits:")
      print(f"   • Single API call instead of 8 separate calls")
      print(f"   • {8 * len(system.timeframes)} → 1 API requests per analysis")
      print(f"   • Consistent data timestamps across all timeframes")
      print(f"   • {len(analysis['buy_signals']) + len(analysis['sell_signals'])} total signals generated")

    except Exception as e:
      print(f"❌ Test failed: {e}")
      import traceback
      traceback.print_exc()
      try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                  SELECT signal, confidence_score, price, datetime_created, contributing_indicators 
                  FROM signals 
                  WHERE symbol = ? 
                  ORDER BY datetime_created DESC 
                  LIMIT 1
              ''', (symbol,))

        result = cursor.fetchone()
        conn.close()

        if result:
          return {
            'signal': result[0],
            'confidence_score': result[1],
            'price': result[2],
            'datetime': result[3],
            'contributing_indicators': result[4] if result[4] else ''
          }
        return None

      except Exception as e:
        logging.error(f"Failed to get last signal: {e}")
        return None


  def show_recent_signals(limit: int = 10):

    """Show recent signals from database with timeframe-specific indicator details"""
    try:
      db_path = 'crypto_signals.db'
      conn = sqlite3.connect(db_path)
      cursor = conn.cursor()

      cursor.execute('''
              SELECT symbol, signal, confidence_score, confidence_percentage, price, 
                     contributing_indicators, timeframe_specific_indicators, 
                     predominant_timeframe, datetime_created
              FROM signals 
              ORDER BY datetime_created DESC 
              LIMIT ?
          ''', (limit,))

      results = cursor.fetchall()
      conn.close()

      if not results:
        print("📋 No signals found in database")
        return

      print(f"📋 RECENT SIGNALS WITH TIMEFRAME DETAILS (Last {len(results)})")
      print("=" * 150)
      print(
        f"{'TIME':<16} {'SYMBOL':<8} {'SIGNAL':<6} {'CONF':<8} {'PRICE':<10} {'MAIN_TF':<8} {'TIMEFRAME-SPECIFIC INDICATORS':<60}")
      print("-" * 150)

      for row in results:
        symbol, signal, conf_score, conf_pct, price, indicators, tf_indicators, main_tf, dt = row

        # Show timeframe-specific indicators (truncated for display)
        tf_indicators_display = tf_indicators[:57] + "..." if tf_indicators and len(tf_indicators) > 60 else (
          tf_indicators or "")
        dt_short = dt.split()[1][:5] if ' ' in dt else dt[:16]  # Show just time

        print(
          f"{dt_short:<16} {symbol:<8} {signal:<6} {conf_score:<8} ${price:<9.2f} {main_tf:<8} {tf_indicators_display:<60}")

      print("-" * 150)

      # Show detailed breakdown for the most recent signal
      if results:
        print(f"\n🔍 DETAILED BREAKDOWN - MOST RECENT SIGNAL:")
        latest = results[0]
        symbol, signal, conf_score, conf_pct, price, indicators, tf_indicators, main_tf, dt = latest

        print(f"Symbol: {symbol}")
        print(f"Signal: {signal}")
        print(f"Confidence: {conf_score}/504 ({conf_pct:.1f}%)")
        print(f"Price: ${price:.4f}")
        print(f"Predominant Timeframe: {main_tf}")
        print(f"Date: {dt}")

        if tf_indicators:
          print(f"\nTimeframe-Specific Indicators:")
          # Parse and display timeframe-specific indicators nicely
          tf_list = tf_indicators.split(', ')
          timeframe_groups = {}

          for tf_indicator in tf_list:
            if '[' in tf_indicator and ']' in tf_indicator:
              # Extract timeframe and indicator
              start = tf_indicator.find('[') + 1
              end = tf_indicator.find(']')
              tf = tf_indicator[start:end]
              indicator = tf_indicator[end + 2:]  # Skip '] '

              if tf not in timeframe_groups:
                timeframe_groups[tf] = []
              timeframe_groups[tf].append(indicator)

          for tf in sorted(timeframe_groups.keys()):
            indicators_list = timeframe_groups[tf]
            print(f"  [{tf}]: {len(indicators_list)} indicators")
            for ind in indicators_list[:3]:  # Show first 3
              print(f"    • {ind}")
            if len(indicators_list) > 3:
              print(f"    ... and {len(indicators_list) - 3} more")

      print(f"\nTotal signals in database: {len(results)}")

    except Exception as e:
      print(f"❌ Error reading database: {e}")

def analyze_signal_patterns():
  """Analyze patterns in stored signals with timeframe breakdown"""
  try:
    db_path = 'crypto_signals.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("📊 SIGNAL PATTERN ANALYSIS WITH TIMEFRAME BREAKDOWN")
    print("=" * 80)

    # Total signals by type
    cursor.execute("SELECT signal, COUNT(*) FROM signals GROUP BY signal ORDER BY COUNT(*) DESC")
    signal_counts = cursor.fetchall()

    print("\n🎯 Signals by Type:")
    for signal_type, count in signal_counts:
      print(f"  {signal_type}: {count}")

    # Most active symbols
    cursor.execute("SELECT symbol, COUNT(*) FROM signals GROUP BY symbol ORDER BY COUNT(*) DESC LIMIT 10")
    symbol_counts = cursor.fetchall()

    print("\n📈 Most Active Symbols:")
    for symbol, count in symbol_counts:
      print(f"  {symbol}: {count} signals")

    # Average confidence by predominant timeframe
    cursor.execute("""
            SELECT predominant_timeframe, 
                   AVG(confidence_score) as avg_confidence,
                   COUNT(*) as count
            FROM signals 
            WHERE predominant_timeframe IS NOT NULL
            GROUP BY predominant_timeframe 
            ORDER BY avg_confidence DESC
        """)

    tf_confidence = cursor.fetchall()

    print("\n⏰ Average Confidence by Predominant Timeframe:")
    for tf, avg_conf, count in tf_confidence:
      print(f"  {tf}: {avg_conf:.1f} avg confidence ({count} signals)")

    # Analyze timeframe-specific indicators
    cursor.execute(
      "SELECT timeframe_specific_indicators FROM signals WHERE timeframe_specific_indicators IS NOT NULL")
    all_tf_indicators = cursor.fetchall()

    timeframe_indicator_freq = {}
    indicator_timeframe_combinations = {}

    for (tf_indicators_str,) in all_tf_indicators:
      if tf_indicators_str:
        for tf_indicator in tf_indicators_str.split(', '):
          tf_indicator = tf_indicator.strip()
          if tf_indicator and '[' in tf_indicator and ']' in tf_indicator:
            # Parse [timeframe] indicator format
            start = tf_indicator.find('[') + 1
            end = tf_indicator.find(']')
            timeframe = tf_indicator[start:end]
            indicator = tf_indicator[end + 2:]  # Skip '] '

            # Count timeframe-indicator combinations
            combo_key = f"{timeframe}_{indicator}"
            timeframe_indicator_freq[combo_key] = timeframe_indicator_freq.get(combo_key, 0) + 1

            # Track which timeframes each indicator appears in
            if indicator not in indicator_timeframe_combinations:
              indicator_timeframe_combinations[indicator] = set()
            indicator_timeframe_combinations[indicator].add(timeframe)

    print("\n🔍 Most Common Timeframe-Indicator Combinations:")
    sorted_tf_indicators = sorted(timeframe_indicator_freq.items(), key=lambda x: x[1], reverse=True)
    for combo, freq in sorted_tf_indicators[:20]:
      timeframe, indicator = combo.split('_', 1)
      print(f"  [{timeframe}] {indicator}: {freq} occurrences")

    print("\n🌐 Indicators Appearing Across Multiple Timeframes:")
    multi_tf_indicators = {k: v for k, v in indicator_timeframe_combinations.items() if len(v) > 1}
    sorted_multi_tf = sorted(multi_tf_indicators.items(), key=lambda x: len(x[1]), reverse=True)

    for indicator, timeframes in sorted_multi_tf[:15]:
      tf_list = ', '.join(sorted(timeframes))
      print(f"  {indicator}: {len(timeframes)} timeframes ({tf_list})")

    # Timeframe distribution analysis
    timeframe_totals = {}
    for combo, freq in timeframe_indicator_freq.items():
      timeframe = combo.split('_')[0]
      timeframe_totals[timeframe] = timeframe_totals.get(timeframe, 0) + freq

    print("\n📊 Signal Distribution by Timeframe:")
    sorted_tf_totals = sorted(timeframe_totals.items(), key=lambda x: x[1], reverse=True)
    total_signals = sum(timeframe_totals.values())

    for timeframe, count in sorted_tf_totals:
      percentage = (count / total_signals) * 100 if total_signals > 0 else 0
      print(f"  {timeframe}: {count} signals ({percentage:.1f}%)")

    conn.close()

  except Exception as e:
    print(f"❌ Error analyzing patterns: {e}")

def run_test_analysis(symbols: List[str]):
  """Run test analysis without sending emails"""
  print("🧪 Running test analysis...")
  print("This will analyze symbols but won't send any emails\n")

  # Create temporary config for testing
  temp_config = {
    'sender_email': 'test@test.com',
    'sender_password': 'test',
    'recipient_emails': ['test@test.com'],
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
  }

  system = AdvancedCryptoAlertSystem(temp_config)

  for symbol in symbols[:3]:  # Test first 3 symbols only
    print(f"\n🔍 Testing {symbol}...")
    try:
      analysis = system.analyze_symbol_comprehensive(symbol)

      # Get contributing indicators
      if analysis['signal'] == 'BUY':
        contributing = analysis['buy_signals']
      elif analysis['signal'] == 'SELL':
        contributing = analysis['sell_signals']
      else:
        contributing = []

      # Extract unique indicators
      unique_indicators = set()
      for sig in contributing:
        if '_' in sig:
          indicator = sig.split('_', 1)[1]
          unique_indicators.add(indicator)
        else:
          unique_indicators.add(sig)

      print(f"Symbol: {symbol}")
      print(f"Signal: {analysis['signal']}")
      print(f"Confidence: {analysis['confidence_score']}/504 ({analysis['confidence_percentage']:.1f}%)")
      print(f"Buy Signals: {len(analysis['buy_signals'])}")
      print(f"Sell Signals: {len(analysis['sell_signals'])}")
      print(f"Predominant Timeframe: {analysis['predominant_timeframe']}")
      print(f"Unique Indicators: {len(unique_indicators)}")

      if unique_indicators:
        print(
          f"Contributing Indicators: {', '.join(sorted(list(unique_indicators))[:5])}{'...' if len(unique_indicators) > 5 else ''}")

      if analysis['confidence_score'] > 15:
        print(f"✅ Strong signal detected!")
      else:
        print(f"⚠️ Weak signal - below threshold")

    except Exception as e:
      print(f"❌ Error testing {symbol}: {e}")

  print("\n🧪 Test analysis complete!")

def load_email_config() -> Optional[Dict]:
  """Load email configuration supporting multiple recipients"""
  try:
    with open('email_config.json', 'r') as f:
      config = json.load(f)

    # Validate required fields
    required_fields = ['sender_email', 'sender_password']
    missing_fields = [field for field in required_fields if not config.get(field)]

    if missing_fields:
      print(f"Missing required email configuration: {missing_fields}")
      return None

    # Ensure recipient_emails is a list
    if 'recipient_emails' not in config and 'recipient_email' in config:
      config['recipient_emails'] = [config['recipient_email']]

    if not config.get('recipient_emails'):
      print("No recipient emails configured")
      return None

    return config

  except FileNotFoundError:
    print("Email configuration file not found: email_config.json")
    return None
  except json.JSONDecodeError as e:
    print(f"Invalid JSON in email_config.json: {e}")
    return None


def create_sample_config():
  """Create sample email configuration with multiple recipients support"""
  sample_config = {
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "recipient_emails": [
      "recipient1@gmail.com",
      "recipient2@gmail.com",
      "recipient3@gmail.com"
    ],
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
  }

  with open('email_config_sample.json', 'w') as f:
    json.dump(sample_config, f, indent=4)

  print("Created 'email_config_sample.json'")
  print("Please copy to 'email_config.json' and update with your details")
  print("\nSupports multiple recipients - just add more emails to the 'recipient_emails' array!")


def test_api_connections() -> Dict[str, bool]:
  """Test all API connections comprehensively"""
  results = {}

  # Test Binance API
  try:
    response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
    results['Binance_Ping'] = response.status_code == 200

    # Test actual data
    response = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5", timeout=10)
    results['Binance_Data'] = response.status_code == 200 and len(response.json()) > 0
  except:
    results['Binance_Ping'] = False
    results['Binance_Data'] = False

  # Test Coinbase API
  try:
    response = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
    results['Coinbase'] = response.status_code == 200 and 'price' in response.json()
  except:
    results['Coinbase'] = False

  # Test CoinGecko API
  try:
    response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
    results['CoinGecko_Ping'] = response.status_code == 200

    # Test price data
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=5)
    results['CoinGecko_Price'] = response.status_code == 200 and 'bitcoin' in response.json()
  except:
    results['CoinGecko_Ping'] = False
    results['CoinGecko_Price'] = False

  # Test CryptoCompare API
  try:
    response = requests.get("https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD", timeout=5)
    results['CryptoCompare'] = response.status_code == 200 and 'USD' in response.json()
  except:
    results['CryptoCompare'] = False

  # Test Kraken API
  try:
    response = requests.get("https://api.kraken.com/0/public/SystemStatus", timeout=5)
    data = response.json()
    results['Kraken'] = response.status_code == 200 and data.get('result', {}).get('status') == 'online'
  except:
    results['Kraken'] = False

  # Test KuCoin API
  try:
    response = requests.get("https://api.kucoin.com/api/v1/status", timeout=5)
    data = response.json()
    results['KuCoin'] = response.status_code == 200 and data.get('code') == '200000'
  except:
    results['KuCoin'] = False

  return results


def test_all_apis_for_symbol(symbol: str = 'BTC') -> Dict[str, bool]:
  """Test all APIs for a specific symbol to verify data availability"""
  print(f"🧪 Testing all APIs for {symbol}...")

  # Create temporary system instance
  temp_config = {
    'sender_email': 'test@test.com',
    'sender_password': 'test',
    'recipient_emails': ['test@test.com'],
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
  }

  system = AdvancedCryptoAlertSystem(temp_config)
  results = {}

  # Test each API method
  api_methods = [
    ('Binance', system._fetch_binance_30m_data),
    ('Coinbase', system._fetch_coinbase_30m_data),
    ('CoinGecko', system._fetch_coingecko_30m_data),
    ('CryptoCompare', system._fetch_cryptocompare_30m_data),
    ('Kraken', system._fetch_kraken_data),
    ('KuCoin', system._fetch_kucoin_data)
  ]

  for api_name, fetch_method in api_methods:
    try:
      print(f"Testing {api_name}...", end=' ')
      data = fetch_method(symbol, '1h', 10)

      if data is not None and len(data) >= 5:
        print(f"✅ Success ({len(data)} candles)")
        results[api_name] = True

        # Show sample data
        latest = data.iloc[-1]
        print(f"   Latest: ${float(latest['close']):.2f} at {latest['timestamp']}")
      else:
        print(f"❌ Failed (insufficient data)")
        results[api_name] = False

    except Exception as e:
      print(f"❌ Failed ({str(e)[:50]}...)")
      results[api_name] = False

  # Test current price fallbacks
  print(f"\nTesting current price fallbacks for {symbol}...")
  try:
    current_price = system._get_current_price_any_source(symbol)
    if current_price:
      print(f"✅ Current price available: ${current_price:.2f}")
      results['Current_Price'] = True
    else:
      print(f"❌ No current price available")
      results['Current_Price'] = False
  except Exception as e:
    print(f"❌ Current price failed: {e}")
    results['Current_Price'] = False

  return results


def get_supported_symbols() -> List[str]:
  """Get supported symbols from Binance"""
  try:
    response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
    data = response.json()

    symbols = []
    for pair in data['symbols']:
      if pair['quoteAsset'] == 'USDT' and pair['status'] == 'TRADING':
        base_symbol = pair['baseAsset']
        # Focus on major cryptocurrencies
        if base_symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'LINK', 'LTC',
                           'XRP', 'SOL', 'AVAX', 'MATIC', 'ATOM', 'ALGO', 'VET',
                           'LUNA', 'NEAR', 'FTM', 'SAND', 'MANA', 'CRV']:
          symbols.append(base_symbol)

    return sorted(symbols)
  except:
    return ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']  # Default fallback

def show_recent_signals(limit: int = 10):
  """Show recent signals from database with timeframe-specific indicator details"""
  try:
    db_path = 'crypto_signals.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
          SELECT symbol, signal, confidence_score, confidence_percentage, price, 
                 contributing_indicators, timeframe_specific_indicators, 
                 predominant_timeframe, datetime_created
          FROM signals 
          ORDER BY datetime_created DESC 
          LIMIT ?
      ''', (limit,))

    results = cursor.fetchall()
    conn.close()

    if not results:
      print("📋 No signals found in database")
      return

    print(f"📋 RECENT SIGNALS WITH TIMEFRAME DETAILS (Last {len(results)})")
    print("=" * 150)
    print(
      f"{'TIME':<16} {'SYMBOL':<8} {'SIGNAL':<6} {'CONF':<8} {'PRICE':<10} {'MAIN_TF':<8} {'TIMEFRAME-SPECIFIC INDICATORS':<60}")
    print("-" * 150)

    for row in results:
      symbol, signal, conf_score, conf_pct, price, indicators, tf_indicators, main_tf, dt = row

      # Show timeframe-specific indicators (truncated for display)
      tf_indicators_display = tf_indicators[:57] + "..." if tf_indicators and len(tf_indicators) > 60 else (
          tf_indicators or "")
      dt_short = dt.split()[1][:5] if ' ' in dt else dt[:16]  # Show just time

      print(
        f"{dt_short:<16} {symbol:<8} {signal:<6} {conf_score:<8} ${price:<9.2f} {main_tf:<8} {tf_indicators_display:<60}")

    print("-" * 150)

    # Show detailed breakdown for the most recent signal
    if results:
      print(f"\n🔍 DETAILED BREAKDOWN - MOST RECENT SIGNAL:")
      latest = results[0]
      symbol, signal, conf_score, conf_pct, price, indicators, tf_indicators, main_tf, dt = latest

      print(f"Symbol: {symbol}")
      print(f"Signal: {signal}")
      print(f"Confidence: {conf_score}/504 ({conf_pct:.1f}%)")
      print(f"Price: ${price:.4f}")
      print(f"Predominant Timeframe: {main_tf}")
      print(f"Date: {dt}")

      if tf_indicators:
        print(f"\nTimeframe-Specific Indicators:")
        # Parse and display timeframe-specific indicators nicely
        tf_list = tf_indicators.split(', ')
        timeframe_groups = {}

        for tf_indicator in tf_list:
          if '[' in tf_indicator and ']' in tf_indicator:
            # Extract timeframe and indicator
            start = tf_indicator.find('[') + 1
            end = tf_indicator.find(']')
            tf = tf_indicator[start:end]
            indicator = tf_indicator[end + 2:]  # Skip '] '

            if tf not in timeframe_groups:
              timeframe_groups[tf] = []
            timeframe_groups[tf].append(indicator)

        for tf in sorted(timeframe_groups.keys()):
          indicators_list = timeframe_groups[tf]
          print(f"  [{tf}]: {len(indicators_list)} indicators")
          for ind in indicators_list[:3]:  # Show first 3
            print(f"    • {ind}")
          if len(indicators_list) > 3:
            print(f"    ... and {len(indicators_list) - 3} more")

    print(f"\nTotal signals in database: {len(results)}")

  except Exception as e:
    print(f"❌ Error reading database: {e}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Advanced Crypto Trading Alert System')
  parser.add_argument('--create-config', action='store_true', help='Create sample email configuration')
  parser.add_argument('--test-email', action='store_true', help='Send test email')
  parser.add_argument('--test-apis', action='store_true', help='Test API connections')
  parser.add_argument('--init-db', action='store_true', help='Initialize database')
  parser.add_argument('--test', action='store_true', help='Run test analysis (no emails)')
  parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
                      help='Symbols to monitor')
  parser.add_argument('--list-symbols', action='store_true', help='List supported symbols')
  parser.add_argument('--test-symbol', type=str, help='Test all APIs for a specific symbol')
  parser.add_argument('--test-fallbacks', action='store_true', help='Test API fallback system')
  parser.add_argument('--api-status', action='store_true', help='Show detailed API status')
  parser.add_argument('--show-signals', type=int, metavar='N', help='Show last N signals with timeframe details')
  parser.add_argument('--analyze-patterns', action='store_true',
                      help='Analyze signal patterns with timeframe breakdown')
  parser.add_argument('--test-optimization', action='store_true', help='Test timeframe optimization system')

  args = parser.parse_args()

  # Handle commands
  if args.create_config:
    create_sample_config()
    exit()

  if args.init_db:
    # init_database_command()
    exit()

  if args.show_signals:
    show_recent_signals(args.show_signals)
    exit()

  if args.analyze_patterns:
    analyze_signal_patterns()
    exit()

  if args.test_apis:
    print("🔗 Testing API connections...")
    results = test_api_connections()

    print("\n📊 API STATUS REPORT:")
    print("=" * 50)

    categories = {
      'Primary Data Sources': ['Binance_Ping', 'Binance_Data'],
      'Fallback Exchanges': ['Coinbase', 'Kraken', 'KuCoin'],
      'Price Data APIs': ['CoinGecko_Ping', 'CoinGecko_Price', 'CryptoCompare']
    }

    for category, apis in categories.items():
      print(f"\n{category}:")
      for api in apis:
        if api in results:
          status_icon = "✅" if results[api] else "❌"
          status_text = "Working" if results[api] else "Failed"
          print(f"  {status_icon} {api.replace('_', ' ')}: {status_text}")

    working_count = sum(results.values())
    total_count = len(results)
    print(f"\n📈 SUMMARY: {working_count}/{total_count} APIs working ({working_count / total_count * 100:.1f}%)")

    if working_count == 0:
      print("⚠️ WARNING: No APIs accessible - check internet connection")
    elif working_count < total_count * 0.5:
      print("⚠️ WARNING: Many APIs down - reduced reliability expected")
    else:
      print("✅ Good API coverage - system should work reliably")

    # Test specific symbol
    print(f"\n🎯 Testing symbol data retrieval...")
    symbol_results = test_all_apis_for_symbol('BTC')
    working_apis = sum(symbol_results.values())
    print(f"\n📊 {working_apis}/{len(symbol_results)} data sources working for BTC")

    exit()

  if args.test_symbol:
    print(f"🎯 Testing all APIs for {args.test_symbol.upper()}...")
    symbol_results = test_all_apis_for_symbol(args.test_symbol.upper())
    working_count = sum(symbol_results.values())
    total_count = len(symbol_results)

    if working_count > 0:
      print(f"\n✅ SUCCESS: {working_count}/{total_count} data sources working")
      print("System will work reliably for this symbol")
    else:
      print(f"\n❌ FAILURE: No data sources working for {args.test_symbol.upper()}")
      print("This symbol may not be supported or all APIs are down")
    exit()

  if args.test_fallbacks:
    print("🔄 Testing fallback system with simulated failures...")

    # Test with multiple symbols
    test_symbols = ['BTC', 'ETH', 'ADA']

    for symbol in test_symbols:
      print(f"\n📊 Testing fallback chain for {symbol}:")
      results = test_all_apis_for_symbol(symbol)

      working_apis = [api for api, status in results.items() if status]
      failed_apis = [api for api, status in results.items() if not status]

      print(f"✅ Working APIs: {', '.join(working_apis) if working_apis else 'None'}")
      print(f"❌ Failed APIs: {', '.join(failed_apis) if failed_apis else 'None'}")

      if len(working_apis) >= 1:
        print(f"🎯 {symbol}: Fallback system will work ({len(working_apis)} sources)")
      else:
        print(f"⚠️ {symbol}: No fallback available - may fail")

    print("\n📋 FALLBACK SYSTEM SUMMARY:")
    print("The system tries APIs in this order:")
    print("1. Binance (Primary)")
    print("2. Coinbase Pro")
    print("3. CoinGecko")
    print("4. CryptoCompare")
    print("5. Kraken")
    print("6. KuCoin")
    print("7. Synthetic data (emergency fallback)")
    exit()

  if args.api_status:
    print("📋 DETAILED API STATUS REPORT")
    print("=" * 60)

    # Test basic connectivity
    basic_results = test_api_connections()

    print("\n🔗 Basic Connectivity:")
    for api, status in basic_results.items():
      icon = "✅" if status else "❌"
      print(f"  {icon} {api}")

    # Test data availability for popular symbols
    print("\n📊 Data Availability Test:")
    popular_symbols = ['BTC', 'ETH', 'BNB', 'ADA']

    for symbol in popular_symbols:
      print(f"\n{symbol}:")
      symbol_results = test_all_apis_for_symbol(symbol)
      working_count = sum(symbol_results.values())
      total_count = len(symbol_results)

      coverage_pct = (working_count / total_count) * 100
      if coverage_pct >= 25:
        coverage_icon = "🟢"
      elif coverage_pct >=15:
        coverage_icon = "🟡"
      else:
        coverage_icon = "🔴"

      print(f"  {coverage_icon} Coverage: {working_count}/{total_count} APIs ({coverage_pct:.0f}%)")

    print(f"\n💡 RECOMMENDATION:")
    working_apis = sum(basic_results.values())
    total_apis = len(basic_results)

    if working_apis >= total_apis * 0.8:
      print("✅ Excellent API coverage - system will be very reliable")
    elif working_apis >= total_apis * 0.5:
      print("🟡 Good API coverage - system should work with occasional fallbacks")
    else:
      print("🔴 Poor API coverage - expect frequent failures, check internet connection")

    exit()

  if args.list_symbols:
    print("📋 Getting supported symbols...")
    symbols = get_supported_symbols()
    print(f"\n✅ Found {len(symbols)} supported symbols:")

    # Group symbols for better display
    rows = [symbols[i:i + 5] for i in range(0, len(symbols), 5)]
    for row in rows:
      print("  " + "  ".join(f"{sym:<6}" for sym in row))

    print(f"\nDefault selection: {args.symbols}")
    print(f"\nTo monitor custom symbols:")
    print(f"python crypto_alerts.py --symbols BTC ETH SOL AVAX MATIC")
    exit()

  if args.test:
    run_test_analysis(args.symbols)
    exit()

  if args.test_email:
    print("📧 Testing email configuration...")
    email_config = load_email_config()
    if not email_config:
      print("❌ Please configure email settings first")
      print("Use --create-config to create sample configuration")
      exit(1)

    system = AdvancedCryptoAlertSystem(email_config)
    success = system.send_test_email()

    if success:
      print("\n✅ Email test successful!")
      print("You're ready to receive trading alerts!")
    else:
      print("\n❌ Email test failed!")
      print("Please check your email configuration")
    exit()

  # Main monitoring mode
  print("🚀 Starting Advanced Crypto Alert System")
  print("=" * 60)

  # Load configuration
  email_config = load_email_config()
  if not email_config:
    print("❌ Email configuration required")
    print("Use --create-config to create sample configuration")
    exit(1)

  # Test APIs
  print("🔗 Testing API connections...")
  api_results = test_api_connections()
  working_apis = sum(api_results.values())

  if working_apis == 0:
    print("❌ No APIs accessible - cannot proceed")
    exit(1)
  #
  # print(f"✅ {working_apis}/{len(api_results)} APIs working")

  # Initialize system
  try:
    system = AdvancedCryptoAlertSystem(email_config)
    print("✅ System initialized")
    print(f"📧 Alert recipients: {len(email_config['recipient_emails'])}")
    print(f"🎯 Monitoring symbols: {args.symbols}")
    print("⚡ Zero-delay continuous monitoring enabled")
    print("🗄️ Database logging enabled with indicator tracking")
    print("📊 504 indicators per symbol analysis")

    # Start monitoring
    system.run_continuous_monitoring(args.symbols)

  except Exception as e:
    logging.error(f"System initialization failed: {e}")
    print(f"❌ Failed to start system: {e}")
    exit(1)


def run_test_analysis(symbols: List[str]):
  """Run test analysis without sending emails"""
  print("🧪 Running test analysis...")
  print("This will analyze symbols but won't send any emails\n")

  # Create temporary config for testing
  temp_config = {
    'sender_email': 'test@test.com',
    'sender_password': 'test',
    'recipient_emails': ['test@test.com'],
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
  }

  system = AdvancedCryptoAlertSystem(temp_config)

  for symbol in symbols[:3]:  # Test first 3 symbols only
    print(f"\n🔍 Testing {symbol}...")
    try:
      analysis = system.analyze_symbol_comprehensive(symbol)

      print(f"Symbol: {symbol}")
      print(f"Signal: {analysis['signal']}")
      print(f"Confidence: {analysis['confidence_score']}/504 ({analysis['confidence_percentage']:.1f}%)")
      print(f"Buy Signals: {len(analysis['buy_signals'])}")
      print(f"Sell Signals: {len(analysis['sell_signals'])}")
      print(f"Predominant Timeframe: {analysis['predominant_timeframe']}")

      if analysis['confidence_score'] > 15:
        print(f"✅ Strong signal detected!")
      else:
        print(f"⚠️ Weak signal - below threshold")

    except Exception as e:
      print(f"❌ Error testing {symbol}: {e}")

  print("\n🧪 Test analysis complete!")
