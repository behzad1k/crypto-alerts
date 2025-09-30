"""
Crypto Trading Signal Accuracy Analyzer - Enhanced Version

Analyzes the performance of trading signals from the crypto_signals.db database.
Evaluates which indicators and timeframes had the best accuracy by checking
if the price moved in the predicted direction within the next 5 candles.

Enhanced Features:
- Buy/Sell signal ratio analysis
- Exclude specific symbols from analysis
- Filter by buy/sell signal ratio thresholds
- 2-3 indicator combinations (same timeframe AND cross-timeframe)
- Filter by signal type (BUY only, SELL only, or both)
- Rich statistical metrics

Usage:
    python signal_analyzer.py --analyze-all
    python signal_analyzer.py --exclude-symbols BTC ETH
    python signal_analyzer.py --signal-type BUY
    python signal_analyzer.py --signal-type SELL
    python signal_analyzer.py --min-buy-sell-ratio 0.3
    python signal_analyzer.py --max-buy-sell-ratio 3.0
    python signal_analyzer.py --min-samples 10
"""

import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

class SignalAccuracyAnalyzer:
    def __init__(self, db_path='crypto_signals.db', candles_ahead=5):
        self.db_path = db_path
        self.candles_ahead = candles_ahead
        self.timeframe_minutes = {
            '30m': 30, '1h': 60, '2h': 120, '4h': 240,
            '6h': 360, '8h': 480, '12h': 720, '1d': 1440
        }

    def get_all_signals(self, exclude_symbols: List[str] = None,
                       signal_type: str = None,
                       min_indicators_per_timeframe: int = None) -> List[Dict]:
        """Retrieve all signals from database with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT id, symbol, signal, confidence_score, confidence_percentage,
                   price, predominant_timeframe, timeframe_specific_indicators,
                   contributing_indicators, datetime_created
            FROM signals
            ORDER BY datetime_created DESC
        '''

        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        exclude_set = set(s.upper() for s in exclude_symbols) if exclude_symbols else set()
        signal_type_filter = signal_type.upper() if signal_type else None

        signals = []
        for row in results:
            symbol = row[1]
            sig_type = row[2]
            tf_indicators_str = row[7]

            if symbol.upper() in exclude_set:
                continue

            if signal_type_filter and sig_type != signal_type_filter:
                continue

            # Filter by minimum indicators per timeframe
            if min_indicators_per_timeframe:
                tf_indicators = self.parse_timeframe_indicators(tf_indicators_str)

                # Check if ANY timeframe has at least min_indicators_per_timeframe unique indicators
                has_sufficient_indicators = False
                for timeframe, indicators in tf_indicators.items():
                    unique_indicators = len(set(indicators))
                    if unique_indicators >= min_indicators_per_timeframe:
                        has_sufficient_indicators = True
                        break

                if not has_sufficient_indicators:
                    continue

            signals.append({
                'id': row[0],
                'symbol': symbol,
                'signal': sig_type,
                'confidence_score': row[3],
                'confidence_percentage': row[4],
                'entry_price': row[5],
                'predominant_timeframe': row[6],
                'timeframe_specific_indicators': tf_indicators_str,
                'contributing_indicators': row[8],
                'datetime_created': datetime.strptime(row[9], '%Y-%m-%d %H:%M:%S')
            })

        return signals

    def get_price_after_signal(self, symbol: str, signal_time: datetime,
                               timeframe: str, candles_ahead: int = 5) -> Optional[float]:
        """Get price after N candles from signal time"""
        minutes_per_candle = self.timeframe_minutes.get(timeframe, 60)
        target_time = signal_time + timedelta(minutes=minutes_per_candle * candles_ahead)

        try:
            symbol_pair = f"{symbol}USDT"
            interval_map = {
                '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
                '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d'
            }

            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol_pair,
                'interval': interval_map.get(timeframe, '1h'),
                'startTime': int(signal_time.timestamp() * 1000),
                'endTime': int((target_time + timedelta(hours=24)).timestamp() * 1000),
                'limit': 20
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) <= candles_ahead:
                return None

            target_candle = data[candles_ahead] if len(data) > candles_ahead else data[-1]
            return float(target_candle[4])

        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None

    def evaluate_signal_success(self, signal: Dict) -> Optional[bool]:
        """Determine if a signal was successful"""
        entry_price = signal['entry_price']
        future_price = self.get_price_after_signal(
            signal['symbol'],
            signal['datetime_created'],
            signal['predominant_timeframe'],
            self.candles_ahead
        )

        if future_price is None:
            return None

        price_change_pct = ((future_price - entry_price) / entry_price) * 100

        if signal['signal'] == 'BUY':
            return price_change_pct > 0
        elif signal['signal'] == 'SELL':
            return price_change_pct < 0

        return None

    def parse_timeframe_indicators(self, tf_indicators_str: str) -> Dict[str, List[str]]:
        """Parse timeframe-specific indicators string"""
        timeframe_indicators = defaultdict(list)

        if not tf_indicators_str:
            return timeframe_indicators

        for item in tf_indicators_str.split(', '):
            if '[' in item and ']' in item:
                start = item.find('[') + 1
                end = item.find(']')
                timeframe = item[start:end]
                indicator = item[end+2:]
                timeframe_indicators[timeframe].append(indicator)

        return timeframe_indicators

    def calculate_buy_sell_ratio(self, stats: Dict) -> float:
        """Calculate buy/sell signal ratio"""
        buy_count = stats.get('buy_signals', 0)
        sell_count = stats.get('sell_signals', 0)

        if sell_count == 0:
            return float('inf') if buy_count > 0 else 0

        return buy_count / sell_count

    def filter_by_ratio(self, data: Dict, min_ratio: float = None,
                       max_ratio: float = None) -> Dict:
        """Filter results by buy/sell ratio thresholds"""
        if min_ratio is None and max_ratio is None:
            return data

        filtered = {}
        for key, stats in data.items():
            ratio = self.calculate_buy_sell_ratio(stats)

            if ratio == float('inf'):
                continue

            if min_ratio is not None and ratio < min_ratio:
                continue

            if max_ratio is not None and ratio > max_ratio:
                continue

            filtered[key] = stats

        return filtered

    def analyze_indicator_combinations(self, signals: List[Dict],
                                      min_samples: int = 5) -> Dict:
        """Analyze accuracy of 2-3 indicator combinations within same timeframe AND cross-timeframe"""
        from itertools import combinations

        combo_stats = {
            'same_timeframe_two': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'same_timeframe_three': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'cross_timeframe_two': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'cross_timeframe_three': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            })
        }

        for signal in signals:
            success = self.evaluate_signal_success(signal)
            if success is None:
                continue

            sig_type = signal['signal']
            status = 'successful' if success else 'failed'

            # Parse timeframe-specific indicators
            tf_indicators = self.parse_timeframe_indicators(
                signal['timeframe_specific_indicators']
            )

            # Helper function to update stats
            def update_stats(stats):
                stats['total'] += 1
                stats[status] += 1

                if sig_type == 'BUY':
                    stats['buy_signals'] += 1
                    if success:
                        stats['buy_successful'] += 1
                    else:
                        stats['buy_failed'] += 1
                elif sig_type == 'SELL':
                    stats['sell_signals'] += 1
                    if success:
                        stats['sell_successful'] += 1
                    else:
                        stats['sell_failed'] += 1

            # SAME TIMEFRAME combinations
            for timeframe, indicators in tf_indicators.items():
                unique_indicators = list(set(indicators))

                # 2-indicator combinations
                if len(unique_indicators) >= 2:
                    for combo in combinations(sorted(unique_indicators), 2):
                        key = f"[{timeframe}] {combo[0]} + {combo[1]}"
                        update_stats(combo_stats['same_timeframe_two'][key])

                # 3-indicator combinations
                if len(unique_indicators) >= 3:
                    for combo in combinations(sorted(unique_indicators), 3):
                        key = f"[{timeframe}] {combo[0]} + {combo[1]} + {combo[2]}"
                        update_stats(combo_stats['same_timeframe_three'][key])

            # CROSS-TIMEFRAME combinations
            # Create list of all timeframe-indicator pairs
            all_tf_indicators = []
            for timeframe, indicators in tf_indicators.items():
                for indicator in set(indicators):
                    all_tf_indicators.append((timeframe, indicator))

            # 2-indicator cross-timeframe combinations
            if len(all_tf_indicators) >= 2:
                for combo in combinations(all_tf_indicators, 2):
                    # Only include if timeframes are different
                    if combo[0][0] != combo[1][0]:
                        key = f"[{combo[0][0]}] {combo[0][1]} + [{combo[1][0]}] {combo[1][1]}"
                        update_stats(combo_stats['cross_timeframe_two'][key])

            # 3-indicator cross-timeframe combinations
            if len(all_tf_indicators) >= 3:
                for combo in combinations(all_tf_indicators, 3):
                    # Only include if at least 2 different timeframes
                    timeframes_in_combo = {combo[0][0], combo[1][0], combo[2][0]}
                    if len(timeframes_in_combo) >= 2:
                        key = f"[{combo[0][0]}] {combo[0][1]} + [{combo[1][0]}] {combo[1][1]} + [{combo[2][0]}] {combo[2][1]}"
                        update_stats(combo_stats['cross_timeframe_three'][key])

        # Filter by minimum samples
        for category in combo_stats:
            combo_stats[category] = {
                k: v for k, v in combo_stats[category].items()
                if v['total'] >= min_samples
            }

        return combo_stats

    def analyze_all_signals(self, min_samples: int = 5,
                          exclude_symbols: List[str] = None,
                          signal_type: str = None,
                          min_buy_sell_ratio: float = None,
                          max_buy_sell_ratio: float = None,
                          min_indicators_per_timeframe: int = None) -> Dict:
        """Comprehensive analysis of all signals"""
        print("Loading signals from database...")
        signals = self.get_all_signals(exclude_symbols, signal_type, min_indicators_per_timeframe)

        if exclude_symbols:
            print(f"Excluded symbols: {', '.join(exclude_symbols)}")

        if signal_type:
            print(f"Filtering for {signal_type} signals only")

        if min_indicators_per_timeframe:
            print(f"Filtering for signals with at least {min_indicators_per_timeframe} unique indicators in any timeframe")

        print(f"Found {len(signals)} signals to analyze\n")

        results = {
            'overall': {
                'total': 0, 'successful': 0, 'failed': 0, 'pending': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            },
            'by_symbol': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'by_timeframe': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'by_indicator': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'by_timeframe_indicator': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            }),
            'by_signal_type': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0
            }),
            'by_confidence_range': defaultdict(lambda: {
                'total': 0, 'successful': 0, 'failed': 0,
                'buy_signals': 0, 'sell_signals': 0,
                'buy_successful': 0, 'buy_failed': 0,
                'sell_successful': 0, 'sell_failed': 0
            })
        }

        for i, signal in enumerate(signals, 1):
            print(f"Analyzing signal {i}/{len(signals)}: {signal['symbol']} {signal['signal']}...", end='\r')

            success = self.evaluate_signal_success(signal)
            sig_type = signal['signal']

            # Track buy/sell counts for overall
            if sig_type == 'BUY':
                results['overall']['buy_signals'] += 1
            elif sig_type == 'SELL':
                results['overall']['sell_signals'] += 1

            if success is None:
                results['overall']['pending'] += 1
                continue

            results['overall']['total'] += 1

            if success:
                results['overall']['successful'] += 1
                status = 'successful'
                if sig_type == 'BUY':
                    results['overall']['buy_successful'] += 1
                elif sig_type == 'SELL':
                    results['overall']['sell_successful'] += 1
            else:
                results['overall']['failed'] += 1
                status = 'failed'
                if sig_type == 'BUY':
                    results['overall']['buy_failed'] += 1
                elif sig_type == 'SELL':
                    results['overall']['sell_failed'] += 1

            # Helper function to update stats
            def update_stats(stats_dict):
                stats_dict['total'] += 1
                stats_dict[status] += 1

                if sig_type == 'BUY':
                    stats_dict['buy_signals'] += 1
                    if success:
                        stats_dict['buy_successful'] += 1
                    else:
                        stats_dict['buy_failed'] += 1
                elif sig_type == 'SELL':
                    stats_dict['sell_signals'] += 1
                    if success:
                        stats_dict['sell_successful'] += 1
                    else:
                        stats_dict['sell_failed'] += 1

            # By symbol
            update_stats(results['by_symbol'][signal['symbol']])

            # By predominant timeframe
            update_stats(results['by_timeframe'][signal['predominant_timeframe']])

            # By signal type (BUY/SELL)
            results['by_signal_type'][sig_type]['total'] += 1
            results['by_signal_type'][sig_type][status] += 1

            # By confidence range
            conf_pct = signal['confidence_percentage']
            if conf_pct < 10:
                conf_range = '0-10%'
            elif conf_pct < 20:
                conf_range = '10-20%'
            elif conf_pct < 30:
                conf_range = '20-30%'
            else:
                conf_range = '30%+'

            update_stats(results['by_confidence_range'][conf_range])

            # Parse timeframe-specific indicators
            tf_indicators = self.parse_timeframe_indicators(
                signal['timeframe_specific_indicators']
            )

            # By individual indicator
            unique_indicators = set()
            for indicators_list in tf_indicators.values():
                unique_indicators.update(indicators_list)

            for indicator in unique_indicators:
                update_stats(results['by_indicator'][indicator])

            # By timeframe-indicator combination
            for timeframe, indicators in tf_indicators.items():
                for indicator in indicators:
                    key = f"{timeframe}_{indicator}"
                    update_stats(results['by_timeframe_indicator'][key])

        print(f"\nAnalysis complete!                                        \n")

        # Analyze indicator combinations
        print("Analyzing 2-3 indicator combinations (same timeframe + cross-timeframe)...")
        combo_results = self.analyze_indicator_combinations(signals, min_samples)
        results['indicator_combinations'] = combo_results

        # Apply ratio filters
        results['by_indicator'] = self.filter_by_ratio(
            {k: v for k, v in results['by_indicator'].items() if v['total'] >= min_samples},
            min_buy_sell_ratio, max_buy_sell_ratio
        )

        results['by_timeframe_indicator'] = self.filter_by_ratio(
            {k: v for k, v in results['by_timeframe_indicator'].items() if v['total'] >= min_samples},
            min_buy_sell_ratio, max_buy_sell_ratio
        )

        results['by_timeframe'] = self.filter_by_ratio(
            dict(results['by_timeframe']),
            min_buy_sell_ratio, max_buy_sell_ratio
        )

        results['by_symbol'] = self.filter_by_ratio(
            dict(results['by_symbol']),
            min_buy_sell_ratio, max_buy_sell_ratio
        )

        results['by_confidence_range'] = self.filter_by_ratio(
            dict(results['by_confidence_range']),
            min_buy_sell_ratio, max_buy_sell_ratio
        )

        return results

    def calculate_accuracy(self, stats: Dict, signal_type: str = None) -> float:
        """Calculate accuracy percentage for overall or specific signal type"""
        if signal_type == 'BUY':
            total = stats.get('buy_signals', 0)
            successful = stats.get('buy_successful', 0)
        elif signal_type == 'SELL':
            total = stats.get('sell_signals', 0)
            successful = stats.get('sell_successful', 0)
        else:
            total = stats.get('total', 0)
            successful = stats.get('successful', 0)

        if total == 0:
            return 0.0
        return (successful / total) * 100

    def print_results(self, results: Dict, min_indicators_per_timeframe: int = None):
        """Print analysis results in formatted tables"""
        print("\n" + "="*100)
        print("SIGNAL ACCURACY ANALYSIS REPORT - ENHANCED")
        print("="*100)

        print(f"\n📊 ANALYSIS PARAMETERS:")
        print(f"  Success Window: {self.candles_ahead} candles ahead")
        if min_indicators_per_timeframe:
            print(f"  Min Indicators Filter: ≥{min_indicators_per_timeframe} unique indicators in any timeframe")

        # Overall statistics
        overall = results['overall']
        buy_sell_ratio = self.calculate_buy_sell_ratio(overall)
        buy_acc = self.calculate_accuracy(overall, 'BUY')
        sell_acc = self.calculate_accuracy(overall, 'SELL')

        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Analyzed: {overall['total']}")
        print(f"  Successful: {overall['successful']} ({self.calculate_accuracy(overall):.1f}%)")
        print(f"  Failed: {overall['failed']}")
        print(f"  Pending (too recent): {overall['pending']}")
        print(f"\n  Buy/Sell Ratio: {buy_sell_ratio:.2f}" if buy_sell_ratio != float('inf') else "\n  Buy/Sell Ratio: Only BUY signals")
        print(f"  Buy Signals: {overall['buy_signals']} (Accuracy: {buy_acc:.1f}%)")
        print(f"  Sell Signals: {overall['sell_signals']} (Accuracy: {sell_acc:.1f}%)")

        # By signal type
        print(f"\nBY SIGNAL TYPE:")
        for sig_type, stats in sorted(results['by_signal_type'].items()):
            acc = self.calculate_accuracy(stats)
            print(f"  {sig_type}: {acc:.1f}% accuracy ({stats['successful']}/{stats['total']})")

        # By confidence range
        print(f"\nBY CONFIDENCE RANGE:")
        conf_order = ['0-10%', '10-20%', '20-30%', '30%+']
        for conf_range in conf_order:
            if conf_range in results['by_confidence_range']:
                stats = results['by_confidence_range'][conf_range]
                acc = self.calculate_accuracy(stats)
                ratio = self.calculate_buy_sell_ratio(stats)
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                buy_acc = self.calculate_accuracy(stats, 'BUY')
                sell_acc = self.calculate_accuracy(stats, 'SELL')
                print(f"  {conf_range}: {acc:.1f}% accuracy ({stats['successful']}/{stats['total']}) | "
                      f"B/S Ratio: {ratio_str} | Buy: {buy_acc:.1f}% | Sell: {sell_acc:.1f}%")

        # By symbol (top 10)
        print(f"\nBY SYMBOL (Top 10 by Accuracy):")
        sorted_symbols = sorted(
            results['by_symbol'].items(),
            key=lambda x: self.calculate_accuracy(x[1]),
            reverse=True
        )[:10]

        print(f"  {'SYMBOL':<8} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7} {'BUY_ACC%':<10} {'SELL_ACC%':<10}")
        print("  " + "-"*60)
        for symbol, stats in sorted_symbols:
            acc = self.calculate_accuracy(stats)
            ratio = self.calculate_buy_sell_ratio(stats)
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
            buy_acc = self.calculate_accuracy(stats, 'BUY')
            sell_acc = self.calculate_accuracy(stats, 'SELL')
            print(f"  {symbol:<8} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7} "
                  f"{buy_acc:<10.1f} {sell_acc:<10.1f}")

        # By timeframe
        print(f"\nBY PREDOMINANT TIMEFRAME:")
        sorted_tf = sorted(
            results['by_timeframe'].items(),
            key=lambda x: self.calculate_accuracy(x[1]),
            reverse=True
        )

        print(f"  {'TF':<6} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7} {'BUY_ACC%':<10} {'SELL_ACC%':<10}")
        print("  " + "-"*60)
        for tf, stats in sorted_tf:
            acc = self.calculate_accuracy(stats)
            ratio = self.calculate_buy_sell_ratio(stats)
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
            buy_acc = self.calculate_accuracy(stats, 'BUY')
            sell_acc = self.calculate_accuracy(stats, 'SELL')
            print(f"  {tf:<6} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7} "
                  f"{buy_acc:<10.1f} {sell_acc:<10.1f}")

        # By indicator (top 20)
        print(f"\nBY INDICATOR (Top 20):")
        sorted_indicators = sorted(
            results['by_indicator'].items(),
            key=lambda x: self.calculate_accuracy(x[1]),
            reverse=True
        )[:20]

        print(f"  {'INDICATOR':<40} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
        print("  " + "-"*70)
        for indicator, stats in sorted_indicators:
            acc = self.calculate_accuracy(stats)
            ratio = self.calculate_buy_sell_ratio(stats)
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
            print(f"  {indicator:<40} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")

        # By timeframe-indicator combination (top 25)
        print(f"\nBY TIMEFRAME-INDICATOR COMBINATION (Top 25):")
        sorted_tf_ind = sorted(
            results['by_timeframe_indicator'].items(),
            key=lambda x: self.calculate_accuracy(x[1]),
            reverse=True
        )[:25]

        print(f"  {'TIMEFRAME-INDICATOR':<50} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
        print("  " + "-"*80)
        for tf_indicator, stats in sorted_tf_ind:
            acc = self.calculate_accuracy(stats)
            ratio = self.calculate_buy_sell_ratio(stats)
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
            tf, indicator = tf_indicator.split('_', 1)
            display_name = f"[{tf}] {indicator}"
            print(f"  {display_name:<50} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")

        # Summary statistics
        print(f"\n" + "="*100)
        print("KEY INSIGHTS:")

        # Best timeframe
        if results['by_timeframe']:
            best_tf = max(results['by_timeframe'].items(),
                         key=lambda x: self.calculate_accuracy(x[1]))
            print(f"  Best Timeframe: {best_tf[0]} ({self.calculate_accuracy(best_tf[1]):.1f}% accuracy)")

        # Best indicator
        if results['by_indicator']:
            best_ind = max(results['by_indicator'].items(),
                          key=lambda x: self.calculate_accuracy(x[1]))
            print(f"  Best Indicator: {best_ind[0]} ({self.calculate_accuracy(best_ind[1]):.1f}% accuracy)")

        # Most balanced ratio
        if results['by_timeframe']:
            balanced = min(results['by_timeframe'].items(),
                          key=lambda x: abs(self.calculate_buy_sell_ratio(x[1]) - 1.0))
            ratio = self.calculate_buy_sell_ratio(balanced[1])
            print(f"  Most Balanced Timeframe: {balanced[0]} (B/S ratio: {ratio:.2f})")

        # Indicator combinations
        if 'indicator_combinations' in results:
            self.print_combination_results(results['indicator_combinations'], min_indicators_per_timeframe)

    def print_combination_results(self, combo_results: Dict, min_indicators_per_timeframe: int = None):
        """Print indicator combination analysis results"""
        print(f"\n" + "="*120)
        print("INDICATOR COMBINATION ANALYSIS")
        if min_indicators_per_timeframe:
            print(f"(Filtered for signals with ≥{min_indicators_per_timeframe} indicators per timeframe)")
        print("="*120)

        # SAME TIMEFRAME - Two indicators
        if combo_results['same_timeframe_two']:
            print(f"\nTOP 2-INDICATOR COMBINATIONS (SAME TIMEFRAME):")
            sorted_two = sorted(
                combo_results['same_timeframe_two'].items(),
                key=lambda x: self.calculate_accuracy(x[1]),
                reverse=True
            )[:30]

            print(f"  {'COMBINATION':<80} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
            print("  " + "-"*110)
            for combo, stats in sorted_two:
                acc = self.calculate_accuracy(stats)
                ratio = self.calculate_buy_sell_ratio(stats)
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                print(f"  {combo:<80} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")
        else:
            print(f"\nNo 2-indicator same-timeframe combinations found (need min samples)")

        # SAME TIMEFRAME - Three indicators
        if combo_results['same_timeframe_three']:
            print(f"\nTOP 3-INDICATOR COMBINATIONS (SAME TIMEFRAME):")
            sorted_three = sorted(
                combo_results['same_timeframe_three'].items(),
                key=lambda x: self.calculate_accuracy(x[1]),
                reverse=True
            )[:30]

            print(f"  {'COMBINATION':<95} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
            print("  " + "-"*125)
            for combo, stats in sorted_three:
                acc = self.calculate_accuracy(stats)
                ratio = self.calculate_buy_sell_ratio(stats)
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                print(f"  {combo:<95} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")
        else:
            print(f"\nNo 3-indicator same-timeframe combinations found (need min samples)")

        # CROSS-TIMEFRAME - Two indicators
        if combo_results['cross_timeframe_two']:
            print(f"\nTOP 2-INDICATOR COMBINATIONS (CROSS-TIMEFRAME):")
            sorted_cross_two = sorted(
                combo_results['cross_timeframe_two'].items(),
                key=lambda x: self.calculate_accuracy(x[1]),
                reverse=True
            )[:30]

            print(f"  {'COMBINATION':<95} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
            print("  " + "-"*125)
            for combo, stats in sorted_cross_two:
                acc = self.calculate_accuracy(stats)
                ratio = self.calculate_buy_sell_ratio(stats)
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                print(f"  {combo:<95} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")
        else:
            print(f"\nNo 2-indicator cross-timeframe combinations found (need min samples)")

        # CROSS-TIMEFRAME - Three indicators
        if combo_results['cross_timeframe_three']:
            print(f"\nTOP 3-INDICATOR COMBINATIONS (CROSS-TIMEFRAME):")
            sorted_cross_three = sorted(
                combo_results['cross_timeframe_three'].items(),
                key=lambda x: self.calculate_accuracy(x[1]),
                reverse=True
            )[:30]

            print(f"  {'COMBINATION':<110} {'ACC%':<7} {'TOTAL':<7} {'B/S':<7}")
            print("  " + "-"*140)
            for combo, stats in sorted_cross_three:
                acc = self.calculate_accuracy(stats)
                ratio = self.calculate_buy_sell_ratio(stats)
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                # Truncate very long combos
                display_combo = combo if len(combo) <= 110 else combo[:107] + "..."
                print(f"  {display_combo:<110} {acc:<7.1f} {stats['total']:<7} {ratio_str:<7}")
        else:
            print(f"\nNo 3-indicator cross-timeframe combinations found (need min samples)")

        # Summary insights for combinations
        print(f"\n" + "="*120)
        print("COMBINATION INSIGHTS:")

        if combo_results['same_timeframe_two']:
            best_same_two = max(combo_results['same_timeframe_two'].items(),
                          key=lambda x: self.calculate_accuracy(x[1]))
            print(f"\n  Best 2-Indicator Same-Timeframe: {best_same_two[0]}")
            print(f"    Accuracy: {self.calculate_accuracy(best_same_two[1]):.1f}%")
            print(f"    Sample Size: {best_same_two[1]['total']}")
            print(f"    B/S Ratio: {self.calculate_buy_sell_ratio(best_same_two[1]):.2f}")

        if combo_results['same_timeframe_three']:
            best_same_three = max(combo_results['same_timeframe_three'].items(),
                           key=lambda x: self.calculate_accuracy(x[1]))
            print(f"\n  Best 3-Indicator Same-Timeframe: {best_same_three[0]}")
            print(f"    Accuracy: {self.calculate_accuracy(best_same_three[1]):.1f}%")
            print(f"    Sample Size: {best_same_three[1]['total']}")
            print(f"    B/S Ratio: {self.calculate_buy_sell_ratio(best_same_three[1]):.2f}")

        if combo_results['cross_timeframe_two']:
            best_cross_two = max(combo_results['cross_timeframe_two'].items(),
                          key=lambda x: self.calculate_accuracy(x[1]))
            print(f"\n  Best 2-Indicator Cross-Timeframe: {best_cross_two[0]}")
            print(f"    Accuracy: {self.calculate_accuracy(best_cross_two[1]):.1f}%")
            print(f"    Sample Size: {best_cross_two[1]['total']}")
            print(f"    B/S Ratio: {self.calculate_buy_sell_ratio(best_cross_two[1]):.2f}")

        if combo_results['cross_timeframe_three']:
            best_cross_three = max(combo_results['cross_timeframe_three'].items(),
                           key=lambda x: self.calculate_accuracy(x[1]))
            print(f"\n  Best 3-Indicator Cross-Timeframe: {best_cross_three[0]}")
            print(f"    Accuracy: {self.calculate_accuracy(best_cross_three[1]):.1f}%")
            print(f"    Sample Size: {best_cross_three[1]['total']}")
            print(f"    B/S Ratio: {self.calculate_buy_sell_ratio(best_cross_three[1]):.2f}")

    def export_results(self, results: Dict, filename: str = 'signal_analysis.json'):
        """Export results to JSON file"""
        # Convert combinations if they exist
        combo_export = {}
        if 'indicator_combinations' in results:
            combo_export = {
                'same_timeframe_two': dict(results['indicator_combinations']['same_timeframe_two']),
                'same_timeframe_three': dict(results['indicator_combinations']['same_timeframe_three']),
                'cross_timeframe_two': dict(results['indicator_combinations']['cross_timeframe_two']),
                'cross_timeframe_three': dict(results['indicator_combinations']['cross_timeframe_three'])
            }

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'overall': results['overall'],
            'by_symbol': dict(results['by_symbol']),
            'by_timeframe': dict(results['by_timeframe']),
            'by_indicator': dict(results['by_indicator']),
            'by_timeframe_indicator': dict(results['by_timeframe_indicator']),
            'by_signal_type': dict(results['by_signal_type']),
            'by_confidence_range': dict(results['by_confidence_range']),
            'indicator_combinations': combo_export
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nResults exported to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze crypto trading signal accuracy')
    parser.add_argument('--analyze-all', action='store_true',
                       help='Analyze all signals in database')
    parser.add_argument('--exclude-symbols', nargs='+',
                       help='Symbols to exclude from analysis (e.g., BTC ETH)')
    parser.add_argument('--signal-type', type=str, choices=['BUY', 'SELL', 'buy', 'sell'],
                       help='Filter by signal type: BUY or SELL')
    parser.add_argument('--min-indicators-per-timeframe', type=int,
                       help='Minimum unique indicators required in any timeframe (e.g., 5 or 6)')
    parser.add_argument('--candles-ahead', type=int, default=5,
                       help='Number of candles ahead to check for price movement (default: 5)')
    parser.add_argument('--min-samples', type=int, default=5,
                       help='Minimum sample size for indicator analysis (default: 5)')
    parser.add_argument('--min-buy-sell-ratio', type=float,
                       help='Minimum buy/sell ratio threshold (e.g., 0.3)')
    parser.add_argument('--max-buy-sell-ratio', type=float,
                       help='Maximum buy/sell ratio threshold (e.g., 3.0)')
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    parser.add_argument('--db-path', type=str, default='crypto_signals.db',
                       help='Path to signals database')

    args = parser.parse_args()

    analyzer = SignalAccuracyAnalyzer(args.db_path, candles_ahead=args.candles_ahead)

    if args.analyze_all:
        results = analyzer.analyze_all_signals(
            min_samples=args.min_samples,
            exclude_symbols=args.exclude_symbols,
            signal_type=args.signal_type,
            min_buy_sell_ratio=args.min_buy_sell_ratio,
            max_buy_sell_ratio=args.max_buy_sell_ratio,
            min_indicators_per_timeframe=args.min_indicators_per_timeframe
        )
        analyzer.print_results(results, args.min_indicators_per_timeframe)

        if args.export:
            analyzer.export_results(results, args.export)
    else:
        print("Use --analyze-all to run analysis")
        print("\nExamples:")
        print("  # Full analysis")
        print("  python signal_analyzer.py --analyze-all")
        print("\n  # Analyze only BUY signals")
        print("  python signal_analyzer.py --analyze-all --signal-type BUY")
        print("\n  # Analyze only SELL signals")
        print("  python signal_analyzer.py --analyze-all --signal-type SELL")
        print("\n  # Exclude symbols and filter ratios")
        print("  python signal_analyzer.py --analyze-all --exclude-symbols BTC ETH --min-buy-sell-ratio 0.3")
        print("\n  # Only signals with 6+ indicators in any timeframe")
        print("  python signal_analyzer.py --analyze-all --min-indicators-per-timeframe 6")
        print("\n  # BUY signals only, with strong indicator convergence")
        print("  python signal_analyzer.py --analyze-all --signal-type BUY --min-indicators-per-timeframe 5")
        print("\n  # Check accuracy for longer holding period (10 candles)")
        print("  python signal_analyzer.py --analyze-all --candles-ahead 10")
        print("\n  # Quick scalp analysis (3 candles)")
        print("  python signal_analyzer.py --analyze-all --candles-ahead 3 --signal-type BUY")
        print("\n  # Require higher sample size")
        print("  python signal_analyzer.py --analyze-all --min-samples 10 --export results.json")