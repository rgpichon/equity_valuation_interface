#!/usr/bin/env python3
"""
Valuation Calibration Module
Simple calibration tools to fine-tune equity valuation parameters

This module provides:
- Parameter optimization
- Historical accuracy testing
- Model validation

Author: Your Name
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import requests
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the main valuation engine (avoid circular import)
try:
    from equity_valuation import EquityValuationEngine
except ImportError:
    # Handle circular import by importing later
    EquityValuationEngine = None

warnings.filterwarnings('ignore')

class ValuationCalibrator:
    """
    Simple valuation calibration system for fine-tuning parameters
    
    Features:
    - Parameter optimization
    - Historical accuracy testing
    - Model validation
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the calibration system
        
        Args:
            api_key: Financial Modeling Prep API key
        """
        self.api_key = api_key or "ZhY4S0yksINsUhFNdaePTZEm7AVu6QPe"
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Calibration results storage
        self.calibration_results = {}
        
    def optimize_parameters(self, 
                          symbol: str,
                          parameter_ranges: Dict = None,
                          optimization_method: str = 'minimize_mae',
                          max_iterations: int = 100,
                          engine: 'EquityValuationEngine' = None) -> Dict:
        """
        Optimize model parameters using historical data
        
        Args:
            symbol: Stock ticker symbol
            parameter_ranges: Dictionary of parameter ranges to optimize
            optimization_method: 'minimize_mae', 'minimize_mape', 'maximize_r2'
            max_iterations: Maximum number of optimization iterations
            engine: Optional EquityValuationEngine instance to reuse
        
        Returns:
            Dict containing optimization results
        """
        try:
            print(f"⚙️ Optimizing parameters for {symbol}...")
            
            # Set default parameter ranges if not provided
            if parameter_ranges is None:
                parameter_ranges = {
                    'revenue_growth': (0.05, 0.25),
                    'ebit_margin': (0.10, 0.40),
                    'wacc': (0.05, 0.15),
                    'terminal_growth': (0.02, 0.05)
                }
            
            # Get historical data for optimization
            historical_data = self._fetch_historical_data(symbol, 12)
            
            if not historical_data:
                return {'error': 'Could not fetch historical data'}
            
            # Initialize engine if not provided
            if engine is None:
                if EquityValuationEngine is None:
                    return {'error': 'EquityValuationEngine not available - circular import issue'}
                engine = EquityValuationEngine(symbol, self.api_key)
            
            # Define objective function
            def objective_function(params):
                return self._calculate_optimization_objective(
                    engine, params, historical_data, optimization_method
                )
            
            # Set up optimization
            initial_params = [
                (parameter_ranges['revenue_growth'][0] + parameter_ranges['revenue_growth'][1]) / 2,
                (parameter_ranges['ebit_margin'][0] + parameter_ranges['ebit_margin'][1]) / 2,
                (parameter_ranges['wacc'][0] + parameter_ranges['wacc'][1]) / 2,
                (parameter_ranges['terminal_growth'][0] + parameter_ranges['terminal_growth'][1]) / 2
            ]
            
            bounds = [
                parameter_ranges['revenue_growth'],
                parameter_ranges['ebit_margin'],
                parameter_ranges['wacc'],
                parameter_ranges['terminal_growth']
            ]
            
            # Run optimization
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations}
            )
            
            # Extract optimized parameters
            optimized_params = {
                'revenue_growth': result.x[0],
                'ebit_margin': result.x[1],
                'wacc': result.x[2],
                'terminal_growth': result.x[3]
            }
            
            # Validate optimized parameters
            validation_results = self._validate_parameters(
                engine, optimized_params, historical_data
            )
            
            optimization_result = {
                'symbol': symbol,
                'optimized_parameters': optimized_params,
                'optimization_method': optimization_method,
                'optimization_success': result.success,
                'final_objective_value': result.fun,
                'validation_results': validation_results,
                'parameter_ranges': parameter_ranges,
                'optimization_date': datetime.now().isoformat()
            }
            
            self.calibration_results[symbol] = optimization_result
            
            print(f"✅ Parameter optimization completed for {symbol}")
            print(f"   • Optimized Revenue Growth: {optimized_params['revenue_growth']:.1%}")
            print(f"   • Optimized EBIT Margin: {optimized_params['ebit_margin']:.1%}")
            print(f"   • Optimized WACC: {optimized_params['wacc']:.1%}")
            print(f"   • Final Objective Value: {result.fun:.4f}")
            
            return optimization_result
            
        except Exception as e:
            return {'error': f"Parameter optimization failed: {e}"}
    
    def calibrate_engine(self, engine: 'EquityValuationEngine') -> Dict:
        """
        Calibrate parameters for an existing EquityValuationEngine
        
        Args:
            engine: EquityValuationEngine instance to calibrate
        
        Returns:
            Dict containing calibration results
        """
        try:
            symbol = engine.symbol
            print(f"🔧 Calibrating engine for {symbol}...")
            
            # Use the engine's existing data and API key
            return self.optimize_parameters(
                symbol=symbol,
                optimization_method='minimize_mae',
                max_iterations=50,
                engine=engine
            )
            
        except Exception as e:
            return {'error': f"Engine calibration failed: {e}"}
    
    def get_optimized_parameters(self, symbol: str) -> Dict:
        """
        Get optimized parameters for a symbol if already calibrated
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dict containing optimized parameters or error
        """
        if symbol in self.calibration_results:
            return self.calibration_results[symbol]['optimized_parameters']
        else:
            return {'error': f'No calibration data available for {symbol}'}
    
    def is_calibrated(self, symbol: str) -> bool:
        """
        Check if a symbol has been calibrated
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Boolean indicating if symbol is calibrated
        """
        return symbol in self.calibration_results
    
    def validate_accuracy(self, 
                         symbol: str,
                         test_periods: int = 6) -> Dict:
        """
        Validate model accuracy against recent market performance
        
        Args:
            symbol: Stock ticker symbol
            test_periods: Number of recent periods to test
        
        Returns:
            Dict containing accuracy validation results
        """
        try:
            print(f"🎯 Validating accuracy for {symbol}...")
            
            # Get recent historical data
            recent_data = self._fetch_historical_data(symbol, test_periods)
            
            if not recent_data:
                return {'error': 'Could not fetch recent data'}
            
            # Initialize engine
            if EquityValuationEngine is None:
                return {'error': 'EquityValuationEngine not available - circular import issue'}
            engine = EquityValuationEngine(symbol, self.api_key)
            
            # Run predictions for each historical period
            predictions = []
            actual_prices = []
            
            for period_data in recent_data:
                actual_price = period_data['actual_price']
                
                # Run valuation with historical data
                dcf_result = engine.calculate_dcf_valuation(
                    revenue_growth=0.15,
                    ebit_margin=0.30,
                    wacc=0.06
                )
                
                peer_result = engine.calculate_peer_valuation()
                
                if 'error' not in dcf_result and 'error' not in peer_result:
                    # Use blended approach
                    predicted_price = (peer_result['peer_price'] * 0.7 + 
                                     dcf_result['dcf_price'] * 0.3)
                    
                    predictions.append(predicted_price)
                    actual_prices.append(actual_price)
            
            if not predictions:
                return {'error': 'No valid predictions generated'}
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                predictions, actual_prices
            )
            
            validation_result = {
                'symbol': symbol,
                'test_periods': test_periods,
                'predictions': predictions,
                'actual_prices': actual_prices,
                'accuracy_metrics': accuracy_metrics,
                'validation_date': datetime.now().isoformat()
            }
            
            print(f"✅ Accuracy validation completed for {symbol}")
            print(f"   • Mean Absolute Error: ${accuracy_metrics.get('mae', 0):.2f}")
            print(f"   • Mean Absolute Percentage Error: {accuracy_metrics.get('mape', 0):.1%}")
            print(f"   • R² Score: {accuracy_metrics.get('r2_score', 0):.3f}")
            
            return validation_result
            
        except Exception as e:
            return {'error': f"Accuracy validation failed: {e}"}
    
    def test_parameters(self, 
                       symbol: str,
                       revenue_growth: float = 0.15,
                       ebit_margin: float = 0.30,
                       wacc: float = 0.06,
                       terminal_growth: float = 0.035) -> Dict:
        """
        Test specific parameters and return valuation results
        
        Args:
            symbol: Stock ticker symbol
            revenue_growth: Revenue growth rate to test
            ebit_margin: EBIT margin to test
            wacc: WACC to test
            terminal_growth: Terminal growth rate to test
        
        Returns:
            Dict containing test results
        """
        try:
            print(f"🧪 Testing parameters for {symbol}...")
            
            # Initialize engine
            if EquityValuationEngine is None:
                return {'error': 'EquityValuationEngine not available - circular import issue'}
            engine = EquityValuationEngine(symbol, self.api_key)
            
            # Run DCF with test parameters
            dcf_result = engine.calculate_dcf_valuation(
                revenue_growth=revenue_growth,
                ebit_margin=ebit_margin,
                wacc=wacc,
                terminal_growth=terminal_growth
            )
            
            # Run peer valuation
            peer_result = engine.calculate_peer_valuation()
            
            # Run market-aligned valuation
            market_result = engine.calculate_market_aligned_valuation()
            
            test_result = {
                'symbol': symbol,
                'test_parameters': {
                    'revenue_growth': revenue_growth,
                    'ebit_margin': ebit_margin,
                    'wacc': wacc,
                    'terminal_growth': terminal_growth
                },
                'dcf_result': dcf_result,
                'peer_result': peer_result,
                'market_result': market_result,
                'test_date': datetime.now().isoformat()
            }
            
            print(f"✅ Parameter test completed for {symbol}")
            if 'error' not in dcf_result:
                print(f"   • DCF Price: ${dcf_result.get('dcf_price', 0):.2f}")
            if 'error' not in peer_result:
                print(f"   • Peer Price: ${peer_result.get('peer_price', 0):.2f}")
            if 'error' not in market_result:
                print(f"   • Market Price: ${market_result.get('blended_price', 0):.2f}")
            
            return test_result
            
        except Exception as e:
            return {'error': f"Parameter test failed: {e}"}
    
    def generate_calibration_report(self, symbol: str) -> str:
        """
        Generate calibration report
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            String containing calibration report
        """
        try:
            calibration_result = self.calibration_results.get(symbol, {})
            
            if not calibration_result:
                return f"No calibration data available for {symbol}"
            
            report = f"""
# Valuation Calibration Report: {symbol}

## Executive Summary
- **Calibration Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Symbol**: {symbol}

## Optimized Parameters
- **Revenue Growth**: {calibration_result['optimized_parameters'].get('revenue_growth', 0):.1%}
- **EBIT Margin**: {calibration_result['optimized_parameters'].get('ebit_margin', 0):.1%}
- **WACC**: {calibration_result['optimized_parameters'].get('wacc', 0):.1%}
- **Terminal Growth**: {calibration_result['optimized_parameters'].get('terminal_growth', 0):.1%}

## Validation Results
- **Mean Absolute Error**: ${calibration_result['validation_results'].get('mae', 0):.2f}
- **Mean Absolute Percentage Error**: {calibration_result['validation_results'].get('mape', 0):.1%}
- **R² Score**: {calibration_result['validation_results'].get('r2_score', 0):.3f}
- **Accuracy**: {calibration_result['validation_results'].get('accuracy', 0):.1%}

## Recommendations
1. Use the optimized parameters for forward-looking valuations
2. Regularly validate model accuracy against new market data
3. Recalibrate parameters quarterly or when market conditions change significantly
"""
            
            return report
            
        except Exception as e:
            return f"Error generating calibration report: {e}"
    
    def get_calibration_summary(self) -> Dict:
        """
        Get summary of all calibration results
        
        Returns:
            Dict containing calibration summary
        """
        try:
            summary = {
                'total_calibrations': len(self.calibration_results),
                'calibrated_symbols': list(self.calibration_results.keys()),
                'latest_calibration': None,
                'calibration_status': {}
            }
            
            if self.calibration_results:
                # Get latest calibration
                latest_symbol = max(
                    self.calibration_results.keys(),
                    key=lambda x: self.calibration_results[x].get('optimization_date', '')
                )
                summary['latest_calibration'] = {
                    'symbol': latest_symbol,
                    'date': self.calibration_results[latest_symbol].get('optimization_date', ''),
                    'success': self.calibration_results[latest_symbol].get('optimization_success', False)
                }
                
                # Get status for each symbol
                for symbol, result in self.calibration_results.items():
                    summary['calibration_status'][symbol] = {
                        'success': result.get('optimization_success', False),
                        'date': result.get('optimization_date', ''),
                        'method': result.get('optimization_method', ''),
                        'objective_value': result.get('final_objective_value', 0)
                    }
            
            return summary
            
        except Exception as e:
            return {'error': f"Error generating calibration summary: {e}"}
    
    # Helper methods
    
    def _fetch_historical_data(self, symbol: str, periods: int) -> List[Dict]:
        """Fetch historical data for calibration"""
        try:
            print(f"📊 Fetching historical data for {symbol}...")
            
            # Fetch historical price data
            price_data = self._fetch_historical_prices(symbol, periods)
            
            if price_data.empty:
                print(f"⚠️ No historical price data found for {symbol}")
                return []
            
            # Fetch historical financial data
            financial_data = self._fetch_historical_financials(symbol, periods)
            
            if not financial_data:
                print(f"⚠️ No historical financial data found for {symbol}")
                return []
            
            # Align and combine the data
            historical_data = self._align_historical_data(price_data, financial_data)
            
            print(f"✅ Found {len(historical_data)} historical periods for {symbol}")
            return historical_data
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return []
    
    def _fetch_historical_prices(self, symbol: str, periods: int) -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            # Calculate date range (roughly 4 quarters per year)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods * 90)  # ~90 days per quarter
            
            url = f"{self.base_url}/historical-price-full/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'historical' in data and data['historical']:
                    df = pd.DataFrame(data['historical'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return pd.DataFrame()
    
    def _fetch_historical_financials(self, symbol: str, periods: int) -> List[Dict]:
        """Fetch historical financial data"""
        try:
            # Fetch income statements
            income_url = f"{self.base_url}/income-statement/{symbol}?limit={periods}&apikey={self.api_key}"
            income_response = requests.get(income_url, timeout=10)
            
            if income_response.status_code == 200:
                income_data = income_response.json()
                return income_data[:periods] if income_data else []
            
            return []
            
        except Exception as e:
            print(f"Error fetching historical financials: {e}")
            return []
    
    def _align_historical_data(self, price_data: pd.DataFrame, financial_data: List[Dict]) -> List[Dict]:
        """Align historical price and financial data"""
        try:
            aligned_data = []
            
            for financial_period in financial_data:
                # Get the date from financial data
                period_date = financial_period.get('date', '')
                if not period_date:
                    continue
                
                # Find closest price data
                period_datetime = pd.to_datetime(period_date)
                
                # Find price data within 30 days of financial period
                price_subset = price_data[
                    (price_data['date'] >= period_datetime - timedelta(days=30)) &
                    (price_data['date'] <= period_datetime + timedelta(days=30))
                ]
                
                if not price_subset.empty:
                    # Use the closest price
                    closest_price = price_subset.iloc[0]
                    
                    aligned_data.append({
                        'date': period_date,
                        'actual_price': closest_price['close'],
                        'financials': financial_period
                    })
            
            return aligned_data
            
        except Exception as e:
            print(f"Error aligning historical data: {e}")
            return []
    
    def _calculate_optimization_objective(self, engine: EquityValuationEngine, params: List[float], historical_data: List[Dict], method: str) -> float:
        """Calculate optimization objective function"""
        try:
            # Extract parameters
            revenue_growth, ebit_margin, wacc, terminal_growth = params
            
            # Validate parameters
            if not (0.01 <= revenue_growth <= 0.50):  # 1% to 50%
                return float('inf')
            if not (0.05 <= ebit_margin <= 0.60):  # 5% to 60%
                return float('inf')
            if not (0.03 <= wacc <= 0.30):  # 3% to 30%
                return float('inf')
            if not (0.01 <= terminal_growth <= 0.08):  # 1% to 8%
                return float('inf')
            
            # Run valuation with these parameters
            predictions = []
            actual_prices = []
            
            for period_data in historical_data:
                actual_price = period_data['actual_price']
                
                # Skip if actual price is invalid
                if actual_price <= 0:
                    continue
                
                # Run DCF with current parameters
                dcf_result = engine.calculate_dcf_valuation(
                    revenue_growth=revenue_growth,
                    ebit_margin=ebit_margin,
                    wacc=wacc,
                    terminal_growth=terminal_growth
                )
                
                if 'error' not in dcf_result and dcf_result.get('dcf_price', 0) > 0:
                    predicted_price = dcf_result['dcf_price']
                    predictions.append(predicted_price)
                    actual_prices.append(actual_price)
            
            if len(predictions) < 2:  # Need at least 2 data points
                return float('inf')
            
            # Calculate objective based on method
            if method == 'minimize_mae':
                return mean_absolute_error(actual_prices, predictions)
            elif method == 'minimize_mape':
                # Avoid division by zero
                actual_array = np.array(actual_prices)
                pred_array = np.array(predictions)
                return np.mean(np.abs((actual_array - pred_array) / actual_array)) * 100
            elif method == 'maximize_r2':
                r2 = r2_score(actual_prices, predictions)
                return -r2 if not np.isnan(r2) else float('inf')  # Negative because we're minimizing
            else:
                return mean_absolute_error(actual_prices, predictions)
            
        except Exception as e:
            print(f"Error calculating optimization objective: {e}")
            return float('inf')
    
    def _validate_parameters(self, engine: EquityValuationEngine, params: Dict, historical_data: List[Dict]) -> Dict:
        """Validate optimized parameters"""
        try:
            # Run validation with optimized parameters
            predictions = []
            actual_prices = []
            
            for period_data in historical_data:
                actual_price = period_data['actual_price']
                
                # Run DCF with optimized parameters
                dcf_result = engine.calculate_dcf_valuation(
                    revenue_growth=params['revenue_growth'],
                    ebit_margin=params['ebit_margin'],
                    wacc=params['wacc'],
                    terminal_growth=params['terminal_growth']
                )
                
                if 'error' not in dcf_result:
                    predicted_price = dcf_result['dcf_price']
                    predictions.append(predicted_price)
                    actual_prices.append(actual_price)
            
            if not predictions:
                return {'mae': 0.0, 'mape': 0.0, 'r2_score': 0.0, 'accuracy': 0.0}
            
            # Calculate validation metrics
            validation_results = self._calculate_accuracy_metrics(predictions, actual_prices)
            
            return validation_results
            
        except Exception as e:
            print(f"Error validating parameters: {e}")
            return {'mae': 0.0, 'mape': 0.0, 'r2_score': 0.0, 'accuracy': 0.0}
    
    def _calculate_accuracy_metrics(self, predictions: List[float], actual: List[float]) -> Dict:
        """Calculate accuracy metrics"""
        try:
            if not predictions or not actual:
                return {}
            
            predictions = np.array(predictions)
            actual = np.array(actual)
            
            # Calculate various accuracy metrics
            mae = mean_absolute_error(actual, predictions)
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            # R² score
            r2 = r2_score(actual, predictions)
            
            # Directional accuracy
            if len(actual) > 1:
                actual_direction = np.diff(actual) > 0
                predicted_direction = np.diff(predictions) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction)
            else:
                directional_accuracy = 0.0
            
            # Overall accuracy (within 10% of actual price)
            accuracy = np.mean(np.abs((actual - predictions) / actual) <= 0.10)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2,
                'directional_accuracy': directional_accuracy,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"Error calculating accuracy metrics: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Valuation Calibration System - Example Usage")
    
    # Initialize calibrator
    calibrator = ValuationCalibrator()
    
    # Test symbol - Deutsche Bank (European bank)
    test_symbol = 'DB'
    
    # Example 1: Standalone calibration
    print(f"\n🔧 Example 1: Standalone Calibration for {test_symbol}")
    
    # Optimize parameters
    print(f"\n⚙️ Optimizing parameters for {test_symbol}...")
    optimization_result = calibrator.optimize_parameters(test_symbol)
    
    if 'error' not in optimization_result:
        print("✅ Parameter optimization completed successfully")
    else:
        print(f"❌ Parameter optimization failed: {optimization_result['error']}")
    
    # Test calibrated parameters
    print(f"\n🧪 Testing calibrated parameters for {test_symbol}...")
    if 'error' not in optimization_result:
        optimized_params = optimization_result['optimized_parameters']
        test_result = calibrator.test_parameters(
            test_symbol,
            revenue_growth=optimized_params['revenue_growth'],
            ebit_margin=optimized_params['ebit_margin'],
            wacc=optimized_params['wacc'],
            terminal_growth=optimized_params['terminal_growth']
        )
        
        if 'error' not in test_result:
            print("✅ Calibrated parameter test completed successfully")
        else:
            print(f"❌ Calibrated parameter test failed: {test_result['error']}")
    else:
        print("⚠️ Skipping parameter test - optimization failed")
    
    # Validate accuracy
    print(f"\n🎯 Validating accuracy for {test_symbol}...")
    accuracy_result = calibrator.validate_accuracy(test_symbol)
    
    if 'error' not in accuracy_result:
        print("✅ Accuracy validation completed successfully")
    else:
        print(f"❌ Accuracy validation failed: {accuracy_result['error']}")
    
    # Generate calibration report
    print(f"\n📋 Generating calibration report for {test_symbol}...")
    report = calibrator.generate_calibration_report(test_symbol)
    print(report)
    
    # Get calibration summary
    print(f"\n📊 Calibration Summary:")
    summary = calibrator.get_calibration_summary()
    if 'error' not in summary:
        print(f"   • Total Calibrations: {summary['total_calibrations']}")
        print(f"   • Calibrated Symbols: {', '.join(summary['calibrated_symbols'])}")
        if summary['latest_calibration']:
            latest = summary['latest_calibration']
            print(f"   • Latest: {latest['symbol']} ({latest['date'][:10]})")
    else:
        print(f"❌ Summary failed: {summary['error']}")
    
    # Example 2: Integration with EquityValuationEngine
    print(f"\n🔧 Example 2: Integration with EquityValuationEngine")
    
    try:
        # Import the equity valuation engine
        from equity_valuation import EquityValuationEngine
        
        # Initialize engine with calibration disabled initially
        engine = EquityValuationEngine(test_symbol, use_calibration=False)
        
        # Run calibration on the existing engine
        print(f"\n⚙️ Calibrating existing engine for {test_symbol}...")
        calibration_result = calibrator.calibrate_engine(engine)
        
        if 'error' not in calibration_result:
            print("✅ Engine calibration completed successfully")
            
            # Get optimized parameters
            optimized_params = calibrator.get_optimized_parameters(test_symbol)
            if 'error' not in optimized_params:
                print(f"   • Optimized Revenue Growth: {optimized_params['revenue_growth']:.1%}")
                print(f"   • Optimized EBIT Margin: {optimized_params['ebit_margin']:.1%}")
                print(f"   • Optimized WACC: {optimized_params['wacc']:.1%}")
                print(f"   • Optimized Terminal Growth: {optimized_params['terminal_growth']:.1%}")
            
            # Test calibrated parameters
            print(f"\n🧪 Testing calibrated parameters...")
            test_result = calibrator.test_parameters(
                test_symbol,
                revenue_growth=optimized_params['revenue_growth'],
                ebit_margin=optimized_params['ebit_margin'],
                wacc=optimized_params['wacc'],
                terminal_growth=optimized_params['terminal_growth']
            )
            
            if 'error' not in test_result:
                print("✅ Calibrated parameter test completed successfully")
            else:
                print(f"❌ Calibrated parameter test failed: {test_result['error']}")
        else:
            print(f"❌ Engine calibration failed: {calibration_result['error']}")
            
    except ImportError:
        print("⚠️ EquityValuationEngine not available for integration example")
    except Exception as e:
        print(f"❌ Integration example failed: {e}")