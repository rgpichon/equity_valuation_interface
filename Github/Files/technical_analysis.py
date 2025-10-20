"""
Technical Analysis Module
Comprehensive technical analysis system with 10 key indicators for stock analysis
Integrates with existing data management system for consistent data access
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from enhanced_comprehensive_data import EnhancedComprehensiveStockData


# ==================== PURE TECHNICAL INDICATOR FUNCTIONS (NO I/O) ====================

def calculate_rsi(ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Pure function with no I/O. Returns RSI values (0-100 scale).
    
    Args:
        ohlcv_df: DataFrame with 'Close' column (OHLC data)
        period: Lookback period for RSI calculation (default 14)
        
    Returns:
        pd.Series with RSI values, index-aligned with input DataFrame
        
    Notes:
        RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    """
    close_prices = ohlcv_df['Close']
    
    # Calculate price changes
    delta = close_prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(ohlcv_df: pd.DataFrame, 
                   fast_period: int = 12, 
                   slow_period: int = 26, 
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Pure function with no I/O. Returns MACD line, signal line, and histogram.
    
    Args:
        ohlcv_df: DataFrame with 'Close' column
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        
    Returns:
        pd.DataFrame with columns: MACD, MACD_Signal, MACD_Histogram
        Index-aligned with input DataFrame
    """
    close_prices = ohlcv_df['Close']
    
    # Calculate EMAs
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })


def calculate_sma(ohlcv_df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA).
    
    Pure function with no I/O. Returns SMA for specified periods.
    
    Args:
        ohlcv_df: DataFrame with 'Close' column
        periods: List of SMA periods to calculate (default [20, 50, 200])
        
    Returns:
        pd.DataFrame with columns: SMA_20, SMA_50, SMA_200, etc.
        Index-aligned with input DataFrame
    """
    close_prices = ohlcv_df['Close']
    
    sma_data = {}
    for period in periods:
        sma_data[f'SMA_{period}'] = close_prices.rolling(window=period).mean()
    
    return pd.DataFrame(sma_data)


def calculate_ema(ohlcv_df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages (EMA).
    
    Pure function with no I/O. Returns EMA for specified periods.
    
    Args:
        ohlcv_df: DataFrame with 'Close' column
        periods: List of EMA periods to calculate (default [12, 26])
        
    Returns:
        pd.DataFrame with columns: EMA_12, EMA_26, etc.
        Index-aligned with input DataFrame
    """
    close_prices = ohlcv_df['Close']
    
    ema_data = {}
    for period in periods:
        ema_data[f'EMA_{period}'] = close_prices.ewm(span=period, adjust=False).mean()
    
    return pd.DataFrame(ema_data)


def calculate_bollinger_bands(ohlcv_df: pd.DataFrame, 
                               period: int = 20, 
                               std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Pure function with no I/O. Returns upper, middle, and lower bands.
    
    Args:
        ohlcv_df: DataFrame with 'Close' column
        period: SMA period for middle band (default 20)
        std_dev: Number of standard deviations for bands (default 2.0)
        
    Returns:
        pd.DataFrame with columns: BB_Upper, BB_Middle, BB_Lower
        Index-aligned with input DataFrame
    """
    close_prices = ohlcv_df['Close']
    
    # Middle band (SMA)
    middle_band = close_prices.rolling(window=period).mean()
    
    # Standard deviation
    std = close_prices.rolling(window=period).std()
    
    # Upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return pd.DataFrame({
        'BB_Upper': upper_band,
        'BB_Middle': middle_band,
        'BB_Lower': lower_band
    })


@dataclass
class TechnicalSignal:
    """Technical analysis signal with strength and confidence"""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: int  # 1-10 scale
    confidence: float  # 0.0-1.0
    indicators: List[str]  # Contributing indicators
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class TechnicalAnalysisEngine:
    """
    Comprehensive technical analysis engine with 10 key indicators
    Phase 1: Moving Averages, MACD, RSI, Bollinger Bands
    Phase 2: Stochastic, ATR, Support/Resistance
    Phase 3: Candlestick Patterns, Multi-Indicator Signals, Risk Management
    """
    
    def __init__(self, data_manager: Optional[EnhancedComprehensiveStockData] = None):
        # Use provided data manager or get shared instance
        self.data_manager = data_manager if data_manager else EnhancedComprehensiveStockData.get_shared_instance(use_cache=True)
        
        # Initialize analysis results storage
        self.analysis_results = {}
        
    def analyze_stock(self, ticker: str) -> Dict:
        """
        Perform complete technical analysis for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete technical analysis results
        """
        print(f"\n🔍 Performing technical analysis for {ticker}...")
        
        try:
            # Get OHLCV data
            ohlcv_data = self.data_manager.export_ohlcv_data(ticker)
            if ohlcv_data is None or ohlcv_data.empty:
                raise ValueError(f"No OHLCV data available for {ticker}")
            
            # Get company name
            company_name = self.data_manager.get_company_name(ticker)
            
            # Perform all technical analysis
            results = {
                'ticker': ticker,
                'company_name': company_name,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(ohlcv_data),
                'latest_price': ohlcv_data['Close'].iloc[-1],
                'latest_date': ohlcv_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            }
            
            # Phase 1: Core Indicators
            results.update(self._analyze_moving_averages(ohlcv_data))
            results.update(self._analyze_macd(ohlcv_data))
            results.update(self._analyze_rsi(ohlcv_data))
            results.update(self._analyze_bollinger_bands(ohlcv_data))
            
            # Phase 2: Intermediate Indicators
            results.update(self._analyze_stochastic(ohlcv_data))
            results.update(self._analyze_atr(ohlcv_data))
            results.update(self._analyze_support_resistance(ohlcv_data))
            
            # Phase 3: Advanced Analysis
            results.update(self._analyze_candlestick_patterns(ohlcv_data))
            results.update(self._generate_multi_indicator_signals(results))
            results.update(self._calculate_risk_management(ohlcv_data, results))
            
            # Store results
            self.analysis_results[ticker] = results
            
            print(f"✅ Technical analysis completed for {ticker}")
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'success': False
            }
    
    # ==================== PHASE 1: CORE INDICATORS ====================
    
    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Enhanced Moving Averages Analysis"""
        close_prices = df['Close']
        
        # Calculate moving averages
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        sma_200 = close_prices.rolling(window=200).mean()
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        
        # Get latest values
        current_price = close_prices.iloc[-1]
        latest_sma_20 = sma_20.iloc[-1]
        latest_sma_50 = sma_50.iloc[-1]
        latest_sma_200 = sma_200.iloc[-1]
        latest_ema_12 = ema_12.iloc[-1]
        latest_ema_26 = ema_26.iloc[-1]
        
        # Trend analysis
        trend_signals = []
        trend_strength = 0
        
        # Price vs MAs
        if current_price > latest_sma_20:
            trend_signals.append("Price > SMA20")
            trend_strength += 1
        if current_price > latest_sma_50:
            trend_signals.append("Price > SMA50")
            trend_strength += 2
        if current_price > latest_sma_200:
            trend_signals.append("Price > SMA200")
            trend_strength += 3
        
        # MA alignment
        ma_alignment = "Bullish" if latest_sma_20 > latest_sma_50 > latest_sma_200 else "Bearish" if latest_sma_20 < latest_sma_50 < latest_sma_200 else "Mixed"
        
        # Golden Cross/Death Cross
        golden_cross = latest_ema_12 > latest_ema_26 and ema_12.iloc[-2] <= ema_26.iloc[-2]
        death_cross = latest_ema_12 < latest_ema_26 and ema_12.iloc[-2] >= ema_26.iloc[-2]
        
        return {
            'moving_averages': {
                'sma_20': latest_sma_20,
                'sma_50': latest_sma_50,
                'sma_200': latest_sma_200,
                'ema_12': latest_ema_12,
                'ema_26': latest_ema_26,
                'trend_signals': trend_signals,
                'trend_strength': trend_strength,
                'ma_alignment': ma_alignment,
                'golden_cross': golden_cross,
                'death_cross': death_cross,
                'primary_trend': "Bullish" if trend_strength >= 4 else "Bearish" if trend_strength <= 1 else "Neutral"
            }
        }
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        """Enhanced MACD Analysis"""
        close_prices = df['Close']
        
        # Calculate MACD
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        # Get latest values
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        latest_histogram = histogram.iloc[-1]
        
        # MACD signals
        macd_signals = []
        macd_strength = 0
        
        # MACD crossover
        if len(macd_line) >= 2:
            if latest_macd > latest_signal and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                macd_signals.append("MACD Bullish Crossover")
                macd_strength += 3
            elif latest_macd < latest_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                macd_signals.append("MACD Bearish Crossover")
                macd_strength -= 3
        
        # Zero line crossover
        if len(macd_line) >= 2:
            if latest_macd > 0 and macd_line.iloc[-2] <= 0:
                macd_signals.append("MACD Above Zero Line")
                macd_strength += 2
            elif latest_macd < 0 and macd_line.iloc[-2] >= 0:
                macd_signals.append("MACD Below Zero Line")
                macd_strength -= 2
        
        # Histogram analysis
        if latest_histogram > 0:
            macd_strength += 1
        else:
            macd_strength -= 1
        
        return {
            'macd': {
                'macd_line': latest_macd,
                'signal_line': latest_signal,
                'histogram': latest_histogram,
                'signals': macd_signals,
                'strength': macd_strength,
                'signal': "Bullish" if macd_strength > 0 else "Bearish" if macd_strength < 0 else "Neutral"
            }
        }
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        """Enhanced RSI Analysis"""
        close_prices = df['Close']
        
        # Calculate RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1]
        
        # RSI signals
        rsi_signals = []
        rsi_strength = 0
        
        # Overbought/Oversold
        if latest_rsi > 70:
            rsi_signals.append("Overbought")
            rsi_strength -= 2
        elif latest_rsi < 30:
            rsi_signals.append("Oversold")
            rsi_strength += 2
        elif 30 <= latest_rsi <= 70:
            rsi_signals.append("Neutral")
        
        # RSI trend
        if len(rsi) >= 10:
            rsi_trend = rsi.iloc[-5:].mean() - rsi.iloc[-10:-5].mean()
            if rsi_trend > 0:
                rsi_signals.append("RSI Rising")
                rsi_strength += 1
            elif rsi_trend < 0:
                rsi_signals.append("RSI Falling")
                rsi_strength -= 1
        
        return {
            'rsi': {
                'value': latest_rsi,
                'signals': rsi_signals,
                'strength': rsi_strength,
                'signal': "Bullish" if rsi_strength > 0 else "Bearish" if rsi_strength < 0 else "Neutral",
                'condition': "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
            }
        }
    
    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Enhanced Bollinger Bands Analysis"""
        close_prices = df['Close']
        
        # Calculate Bollinger Bands
        sma_20 = close_prices.rolling(window=20).mean()
        std_20 = close_prices.rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        # Get latest values
        current_price = close_prices.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_middle = sma_20.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        
        # Band width (volatility measure)
        band_width = (latest_upper - latest_lower) / latest_middle
        
        # Price position
        bb_position = (current_price - latest_lower) / (latest_upper - latest_lower)
        
        # Bollinger Bands signals
        bb_signals = []
        bb_strength = 0
        
        # Price position signals
        if current_price > latest_upper:
            bb_signals.append("Price Above Upper Band")
            bb_strength -= 2
        elif current_price < latest_lower:
            bb_signals.append("Price Below Lower Band")
            bb_strength += 2
        elif latest_lower <= current_price <= latest_upper:
            bb_signals.append("Price Within Bands")
        
        # Band squeeze/expansion
        if band_width < 0.1:
            bb_signals.append("Band Squeeze (Low Volatility)")
        elif band_width > 0.2:
            bb_signals.append("Band Expansion (High Volatility)")
        
        return {
            'bollinger_bands': {
                'upper_band': latest_upper,
                'middle_band': latest_middle,
                'lower_band': latest_lower,
                'band_width': band_width,
                'bb_position': bb_position,
                'signals': bb_signals,
                'strength': bb_strength,
                'signal': "Bullish" if bb_strength > 0 else "Bearish" if bb_strength < 0 else "Neutral",
                'volatility': "Low" if band_width < 0.1 else "High" if band_width > 0.2 else "Normal"
            }
        }
    
    # ==================== PHASE 2: INTERMEDIATE INDICATORS ====================
    
    def _analyze_stochastic(self, df: pd.DataFrame) -> Dict:
        """Stochastic Oscillator Analysis"""
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        
        # Calculate Stochastic
        lowest_low = low_prices.rolling(window=14).min()
        highest_high = high_prices.rolling(window=14).max()
        
        k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        # Get latest values
        latest_k = k_percent.iloc[-1]
        latest_d = d_percent.iloc[-1]
        
        # Stochastic signals
        stoch_signals = []
        stoch_strength = 0
        
        # Overbought/Oversold
        if latest_k > 80:
            stoch_signals.append("Overbought")
            stoch_strength -= 2
        elif latest_k < 20:
            stoch_signals.append("Oversold")
            stoch_strength += 2
        else:
            stoch_signals.append("Neutral")
        
        # K/D crossover
        if len(k_percent) >= 2:
            if latest_k > latest_d and k_percent.iloc[-2] <= d_percent.iloc[-2]:
                stoch_signals.append("Bullish Crossover")
                stoch_strength += 2
            elif latest_k < latest_d and k_percent.iloc[-2] >= d_percent.iloc[-2]:
                stoch_signals.append("Bearish Crossover")
                stoch_strength -= 2
        
        return {
            'stochastic': {
                'k_percent': latest_k,
                'd_percent': latest_d,
                'signals': stoch_signals,
                'strength': stoch_strength,
                'signal': "Bullish" if stoch_strength > 0 else "Bearish" if stoch_strength < 0 else "Neutral",
                'condition': "Overbought" if latest_k > 80 else "Oversold" if latest_k < 20 else "Neutral"
            }
        }
    
    def _analyze_atr(self, df: pd.DataFrame) -> Dict:
        """Average True Range Analysis"""
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        
        # Calculate True Range
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=14).mean()
        
        # Get latest values
        latest_atr = atr.iloc[-1]
        current_price = close_prices.iloc[-1]
        
        # ATR-based signals
        atr_signals = []
        
        # Volatility assessment
        atr_percentage = (latest_atr / current_price) * 100
        
        if atr_percentage > 3:
            volatility_level = "High"
            atr_signals.append("High Volatility")
        elif atr_percentage < 1:
            volatility_level = "Low"
            atr_signals.append("Low Volatility")
        else:
            volatility_level = "Normal"
            atr_signals.append("Normal Volatility")
        
        # Stop-loss calculations
        stop_loss_long = current_price - (2 * latest_atr)
        stop_loss_short = current_price + (2 * latest_atr)
        
        return {
            'atr': {
                'value': latest_atr,
                'atr_percentage': atr_percentage,
                'volatility_level': volatility_level,
                'signals': atr_signals,
                'stop_loss_long': stop_loss_long,
                'stop_loss_short': stop_loss_short,
                'risk_assessment': "High" if atr_percentage > 3 else "Low" if atr_percentage < 1 else "Medium"
            }
        }
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Support and Resistance Level Detection"""
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        # Look for swing points (simplified approach)
        for i in range(2, len(df) - 2):
            # Swing high
            if (high_prices.iloc[i] > high_prices.iloc[i-1] and 
                high_prices.iloc[i] > high_prices.iloc[i-2] and
                high_prices.iloc[i] > high_prices.iloc[i+1] and
                high_prices.iloc[i] > high_prices.iloc[i+2]):
                swing_highs.append(high_prices.iloc[i])
            
            # Swing low
            if (low_prices.iloc[i] < low_prices.iloc[i-1] and 
                low_prices.iloc[i] < low_prices.iloc[i-2] and
                low_prices.iloc[i] < low_prices.iloc[i+1] and
                low_prices.iloc[i] < low_prices.iloc[i+2]):
                swing_lows.append(low_prices.iloc[i])
        
        # Find significant levels (clustering)
        current_price = close_prices.iloc[-1]
        
        # Simple support/resistance detection
        resistance_levels = []
        support_levels = []
        
        if swing_highs:
            # Find resistance levels above current price
            resistance_candidates = [h for h in swing_highs if h > current_price]
            if resistance_candidates:
                resistance_levels = sorted(resistance_candidates)[:3]  # Top 3
        
        if swing_lows:
            # Find support levels below current price
            support_candidates = [l for l in swing_lows if l < current_price]
            if support_candidates:
                support_levels = sorted(support_candidates, reverse=True)[:3]  # Top 3
        
        # Calculate distances
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        nearest_support = max(support_levels) if support_levels else None
        
        return {
            'support_resistance': {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
                'resistance_distance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
            }
        }
    
    # ==================== PHASE 3: ADVANCED ANALYSIS ====================
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Candlestick Pattern Recognition"""
        open_prices = df['Open']
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        
        # Get last few candles for pattern recognition
        patterns = []
        pattern_strength = 0
        
        if len(df) >= 3:
            # Get recent candles
            o1, o2, o3 = open_prices.iloc[-3], open_prices.iloc[-2], open_prices.iloc[-1]
            h1, h2, h3 = high_prices.iloc[-3], high_prices.iloc[-2], high_prices.iloc[-1]
            l1, l2, l3 = low_prices.iloc[-3], low_prices.iloc[-2], low_prices.iloc[-1]
            c1, c2, c3 = close_prices.iloc[-3], close_prices.iloc[-2], close_prices.iloc[-1]
            
            # Doji pattern
            body_size = abs(c2 - o2)
            candle_range = h2 - l2
            if candle_range > 0 and body_size < (candle_range * 0.1):
                patterns.append("Doji")
                pattern_strength += 1
            
            # Hammer pattern
            if (l2 < min(o2, c2) and 
                (h2 - max(o2, c2)) < (min(o2, c2) - l2) * 0.3):
                patterns.append("Hammer")
                pattern_strength += 2
            
            # Shooting Star pattern
            if (h2 > max(o2, c2) and 
                (min(o2, c2) - l2) < (h2 - max(o2, c2)) * 0.3):
                patterns.append("Shooting Star")
                pattern_strength -= 2
            
            # Engulfing patterns
            if len(df) >= 2:
                # Bullish Engulfing
                if (c2 > o2 and c1 < o1 and c2 > o1 and o2 < c1):
                    patterns.append("Bullish Engulfing")
                    pattern_strength += 3
                
                # Bearish Engulfing
                elif (c2 < o2 and c1 > o1 and c2 < o1 and o2 > c1):
                    patterns.append("Bearish Engulfing")
                    pattern_strength -= 3
        
        return {
            'candlestick_patterns': {
                'patterns': patterns,
                'pattern_strength': pattern_strength,
                'signal': "Bullish" if pattern_strength > 0 else "Bearish" if pattern_strength < 0 else "Neutral",
                'reliability': "High" if abs(pattern_strength) >= 3 else "Medium" if abs(pattern_strength) >= 2 else "Low"
            }
        }
    
    def _generate_multi_indicator_signals(self, results: Dict) -> Dict:
        """Generate Multi-Indicator Trading Signals"""
        
        # Collect all indicator strengths
        indicator_strengths = []
        contributing_indicators = []
        
        # Moving Averages
        ma_strength = results['moving_averages']['trend_strength']
        if ma_strength != 0:
            indicator_strengths.append(ma_strength)
            contributing_indicators.append("Moving Averages")
        
        # MACD
        macd_strength = results['macd']['strength']
        if macd_strength != 0:
            indicator_strengths.append(macd_strength)
            contributing_indicators.append("MACD")
        
        # RSI
        rsi_strength = results['rsi']['strength']
        if rsi_strength != 0:
            indicator_strengths.append(rsi_strength)
            contributing_indicators.append("RSI")
        
        # Bollinger Bands
        bb_strength = results['bollinger_bands']['strength']
        if bb_strength != 0:
            indicator_strengths.append(bb_strength)
            contributing_indicators.append("Bollinger Bands")
        
        # Stochastic
        stoch_strength = results['stochastic']['strength']
        if stoch_strength != 0:
            indicator_strengths.append(stoch_strength)
            contributing_indicators.append("Stochastic")
        
        # Candlestick Patterns
        pattern_strength = results['candlestick_patterns']['pattern_strength']
        if pattern_strength != 0:
            indicator_strengths.append(pattern_strength)
            contributing_indicators.append("Candlestick Patterns")
        
        # Calculate overall signal
        if not indicator_strengths:
            overall_signal = "HOLD"
            signal_strength = 5
            confidence = 0.5
        else:
            total_strength = sum(indicator_strengths)
            signal_strength = max(1, min(10, 5 + total_strength))
            
            if total_strength > 2:
                overall_signal = "BUY"
                confidence = min(1.0, 0.5 + (total_strength * 0.1))
            elif total_strength < -2:
                overall_signal = "SELL"
                confidence = min(1.0, 0.5 + (abs(total_strength) * 0.1))
            else:
                overall_signal = "HOLD"
                confidence = 0.5
        
        # Risk level assessment
        atr_risk = results['atr']['risk_assessment']
        volatility = results['bollinger_bands']['volatility']
        
        if atr_risk == "High" or volatility == "High":
            risk_level = "HIGH"
        elif atr_risk == "Low" and volatility == "Low":
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        return {
            'multi_indicator_signals': {
                'overall_signal': overall_signal,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'contributing_indicators': contributing_indicators,
                'indicator_count': len(contributing_indicators),
                'risk_level': risk_level,
                'signal_details': {
                    'ma_strength': ma_strength,
                    'macd_strength': macd_strength,
                    'rsi_strength': rsi_strength,
                    'bb_strength': bb_strength,
                    'stoch_strength': stoch_strength,
                    'pattern_strength': pattern_strength
                }
            }
        }
    
    def _calculate_risk_management(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Risk Management Calculations"""
        current_price = df['Close'].iloc[-1]
        atr = results['atr']['value']
        
        # Stop-loss calculations
        stop_loss_long = results['atr']['stop_loss_long']
        stop_loss_short = results['atr']['stop_loss_short']
        
        # Position sizing (simplified)
        risk_per_trade = 0.02  # 2% risk per trade
        account_size = 100000  # Example account size
        
        # Calculate position size based on ATR stop-loss
        risk_amount = account_size * risk_per_trade
        stop_distance = abs(current_price - stop_loss_long)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Risk-reward ratio
        target_price = current_price * 1.1  # 10% target (simplified)
        reward_distance = target_price - current_price
        risk_reward_ratio = reward_distance / stop_distance if stop_distance > 0 else 0
        
        # Maximum drawdown (simplified calculation)
        rolling_max = df['Close'].expanding().max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'risk_management': {
                'stop_loss_long': stop_loss_long,
                'stop_loss_short': stop_loss_short,
                'target_price': target_price,
                'position_size': position_size,
                'risk_per_trade': risk_per_trade,
                'risk_reward_ratio': risk_reward_ratio,
                'max_drawdown': max_drawdown,
                'risk_amount': risk_amount,
                'stop_distance': stop_distance,
                'reward_distance': reward_distance
            }
        }
    
    # ==================== ANALYSIS METHODS ====================
    
    def analyze_all_stocks(self) -> Dict[str, Dict]:
        """Analyze all available stocks"""
        results = {}
        
        for ticker in self.data_manager.get_stock_list():
            results[ticker] = self.analyze_stock(ticker)
        
        return results
    
    def get_analysis_summary(self, ticker: str) -> Dict:
        """Get summary of technical analysis for a stock"""
        if ticker not in self.analysis_results:
            self.analyze_stock(ticker)
        
        return self.analysis_results.get(ticker, {})
    
    def print_technical_analysis(self, ticker: str, detailed: bool = True):
        """Print formatted technical analysis results"""
        
        if ticker not in self.analysis_results:
            self.analyze_stock(ticker)
        
        results = self.analysis_results.get(ticker, {})
        
        if 'error' in results:
            print(f"❌ Error analyzing {ticker}: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"TECHNICAL ANALYSIS: {results['ticker']} - {results['company_name']}")
        print(f"{'='*80}")
        print(f"Analysis Date: {results['analysis_date']}")
        print(f"Latest Price: ${results['latest_price']:.2f}")
        print(f"Data Points: {results['data_points']}")
        
        # Phase 1: Core Indicators
        print(f"\n--- TREND ANALYSIS ---")
        ma = results['moving_averages']
        print(f"Primary Trend: {ma['primary_trend']}")
        print(f"MA Alignment: {ma['ma_alignment']}")
        print(f"Trend Strength: {ma['trend_strength']}/6")
        if ma['golden_cross']:
            print("🟢 Golden Cross Detected")
        if ma['death_cross']:
            print("🔴 Death Cross Detected")
        
        print(f"\n--- MOMENTUM INDICATORS ---")
        macd = results['macd']
        rsi = results['rsi']
        stoch = results['stochastic']
        
        print(f"MACD: {macd['signal']} (Strength: {macd['strength']})")
        print(f"RSI: {rsi['value']:.1f} ({rsi['condition']}) - {rsi['signal']}")
        print(f"Stochastic: K={stoch['k_percent']:.1f}, D={stoch['d_percent']:.1f} ({stoch['condition']})")
        
        print(f"\n--- VOLATILITY & SUPPORT/RESISTANCE ---")
        bb = results['bollinger_bands']
        atr = results['atr']
        sr = results['support_resistance']
        
        print(f"Bollinger Bands: {bb['volatility']} volatility")
        print(f"ATR: {atr['atr_percentage']:.2f}% ({atr['volatility_level']})")
        if sr['nearest_support']:
            print(f"Nearest Support: ${sr['nearest_support']:.2f} ({sr['support_distance']:.1f}% away)")
        if sr['nearest_resistance']:
            print(f"Nearest Resistance: ${sr['nearest_resistance']:.2f} ({sr['resistance_distance']:.1f}% away)")
        
        print(f"\n--- PATTERN RECOGNITION ---")
        patterns = results['candlestick_patterns']
        if patterns['patterns']:
            print(f"Candlestick Patterns: {', '.join(patterns['patterns'])}")
            print(f"Pattern Reliability: {patterns['reliability']}")
        else:
            print("No significant candlestick patterns detected")
        
        print(f"\n--- TRADING SIGNALS ---")
        signals = results['multi_indicator_signals']
        print(f"Overall Signal: {signals['overall_signal']}")
        print(f"Signal Strength: {signals['signal_strength']}/10")
        print(f"Confidence: {signals['confidence']:.1%}")
        print(f"Risk Level: {signals['risk_level']}")
        print(f"Contributing Indicators: {signals['indicator_count']}")
        
        print(f"\n--- RISK MANAGEMENT ---")
        risk = results['risk_management']
        print(f"Stop Loss (Long): ${risk['stop_loss_long']:.2f}")
        print(f"Target Price: ${risk['target_price']:.2f}")
        print(f"Risk-Reward Ratio: 1:{risk['risk_reward_ratio']:.2f}")
        print(f"Position Size: {risk['position_size']:.0f} shares")
        print(f"Max Drawdown: {risk['max_drawdown']:.1%}")
        
        if detailed:
            print(f"\n--- DETAILED INDICATOR VALUES ---")
            print(f"SMA 20: ${ma['sma_20']:.2f}")
            print(f"SMA 50: ${ma['sma_50']:.2f}")
            print(f"SMA 200: ${ma['sma_200']:.2f}")
            print(f"EMA 12: ${ma['ema_12']:.2f}")
            print(f"EMA 26: ${ma['ema_26']:.2f}")
            print(f"MACD Line: {macd['macd_line']:.4f}")
            print(f"MACD Signal: {macd['signal_line']:.4f}")
            print(f"MACD Histogram: {macd['histogram']:.4f}")
            print(f"BB Upper: ${bb['upper_band']:.2f}")
            print(f"BB Middle: ${bb['middle_band']:.2f}")
            print(f"BB Lower: ${bb['lower_band']:.2f}")
            print(f"ATR: ${atr['value']:.2f}")
        
        print(f"\n{'='*80}\n")


def main():
    """Main function to demonstrate technical analysis"""
    
    print("🔍 Initializing Technical Analysis Engine...")
    
    # Initialize technical analysis engine
    ta_engine = TechnicalAnalysisEngine()
    
    # Analyze all stocks
    print("\n📊 Analyzing all stocks...")
    results = ta_engine.analyze_all_stocks()
    
    # Print results for each stock
    for ticker in ta_engine.data_manager.get_stock_list():
        ta_engine.print_technical_analysis(ticker, detailed=False)
    
    return ta_engine


if __name__ == "__main__":
    technical_analysis = main()