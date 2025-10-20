"""
Equity Derivatives Pricing Project
A comprehensive system for options pricing, strategy construction, and risk management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataLoader:
    """Module 1: Data Loader for equity price data, volatility, and options chains"""
    
    def __init__(self):
        self.price_data = {}
        self.options_data = {}
        self.events_calendar = {}
        
    def load_stock_data(self, symbol, start_date, end_date=None):
        """Load historical stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Calculate daily returns and volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            self.price_data[symbol] = data
            print(f"Loaded {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None
    
    def load_options_chain(self, symbol, expiration_date=None):
        """Load options chain data (synthetic for demonstration)"""
        try:
            ticker = yf.Ticker(symbol)
            # Try to get options data, but fall back to synthetic if not available
            try:
                if expiration_date:
                    chain = ticker.options_chain(expiration_date)
                else:
                    # Get next available expiration
                    expirations = ticker.options
                    if expirations:
                        chain = ticker.options_chain(expirations[0])
                    else:
                        return self._generate_synthetic_options(symbol)
                
                # Combine calls and puts
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls['Type'] = 'Call'
                puts['Type'] = 'Put'
                
                options_data = pd.concat([calls, puts], ignore_index=True)
                self.options_data[symbol] = options_data
                return options_data
                
            except AttributeError:
                # yfinance API changed, use synthetic data
                print(f"Using synthetic options data for {symbol}")
                return self._generate_synthetic_options(symbol)
            
        except Exception as e:
            print(f"Error loading options for {symbol}: {e}")
            return self._generate_synthetic_options(symbol)
    
    def _generate_synthetic_options(self, symbol):
        """Generate synthetic options data for demonstration"""
        # Get current price from loaded data if available, otherwise use default
        if symbol in self.price_data:
            current_price = self.price_data[symbol]['Close'].iloc[-1]
        else:
            current_price = 100  # Default price
        
        strikes = np.arange(current_price * 0.7, current_price * 1.3, 5)
        expirations = [30, 60, 90]  # Days to expiration
        
        options_list = []
        
        for dte in expirations:
            for strike in strikes:
                # Calculate synthetic IV based on moneyness and time
                moneyness = np.log(strike / current_price)
                iv_base = 0.25 + 0.1 * abs(moneyness) + 0.05 * (30 / dte)
                
                # Calls
                call_iv = iv_base + np.random.normal(0, 0.02)
                call_price = self._black_scholes(current_price, strike, dte/365, 0.02, call_iv, 'call')
                
                options_list.append({
                    'Strike': strike,
                    'Expiration': dte,
                    'Type': 'Call',
                    'IV': call_iv,
                    'Bid': call_price * 0.98,
                    'Ask': call_price * 1.02,
                    'Volume': np.random.randint(10, 1000)
                })
                
                # Puts
                put_iv = iv_base + np.random.normal(0, 0.02)
                put_price = self._black_scholes(current_price, strike, dte/365, 0.02, put_iv, 'put')
                
                options_list.append({
                    'Strike': strike,
                    'Expiration': dte,
                    'Type': 'Put',
                    'IV': put_iv,
                    'Bid': put_price * 0.98,
                    'Ask': put_price * 1.02,
                    'Volume': np.random.randint(10, 1000)
                })
        
        options_df = pd.DataFrame(options_list)
        # Store the synthetic options data
        self.options_data[symbol] = options_df
        return options_df
    
    def _black_scholes(self, S, K, T, r, sigma, option_type):
        """Basic Black-Scholes implementation"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return price
    
    def add_event(self, symbol, event_date, event_type, description=""):
        """Add trading event to calendar"""
        if symbol not in self.events_calendar:
            self.events_calendar[symbol] = []
        
        self.events_calendar[symbol].append({
            'date': event_date,
            'type': event_type,
            'description': description
        })
    
    def get_events(self, symbol, start_date=None, end_date=None):
        """Get events for a symbol within date range"""
        if symbol not in self.events_calendar:
            return []
        
        events = self.events_calendar[symbol]
        if start_date and end_date:
            events = [e for e in events if start_date <= e['date'] <= end_date]
        
        return events


class VolatilityModel:
    """Module 2: Implied Move and Volatility Model"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def calculate_atm_straddle_iv(self, symbol, expiration_days):
        """Calculate ATM straddle implied volatility"""
        if symbol not in self.data_loader.options_data:
            return None
        
        options = self.data_loader.options_data[symbol]
        current_price = self.data_loader.price_data[symbol]['Close'].iloc[-1]
        
        # Find ATM options
        atm_options = options[
            (options['Strike'] >= current_price * 0.95) & 
            (options['Strike'] <= current_price * 1.05)
        ]
        
        if len(atm_options) == 0:
            return None
        
        # Calculate straddle IV
        atm_calls = atm_options[atm_options['Type'] == 'Call']
        atm_puts = atm_options[atm_options['Type'] == 'Put']
        
        if len(atm_calls) > 0 and len(atm_puts) > 0:
            call_iv = atm_calls['IV'].iloc[0]
            put_iv = atm_puts['IV'].iloc[0]
            straddle_iv = (call_iv + put_iv) / 2
            return straddle_iv
        
        return None
    
    def estimate_implied_move(self, symbol, expiration_days):
        """Estimate implied move using ATM straddle pricing"""
        straddle_iv = self.calculate_atm_straddle_iv(symbol, expiration_days)
        if straddle_iv is None:
            return None
        
        # Implied move = IV * sqrt(time) * current_price
        current_price = self.data_loader.price_data[symbol]['Close'].iloc[-1]
        time_factor = np.sqrt(expiration_days / 365)
        implied_move = straddle_iv * time_factor * current_price
        
        return implied_move
    
    def calculate_historical_volatility(self, symbol, window=20):
        """Calculate historical realized volatility"""
        if symbol not in self.data_loader.price_data:
            return None
        
        data = self.data_loader.price_data[symbol]
        returns = data['Returns'].dropna()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Pre/post event volatility (if events exist)
        events = self.data_loader.get_events(symbol)
        event_vols = {}
        
        for event in events:
            event_date = event['date']
            if event_date in data.index:
                # Pre-event volatility (10 days before)
                pre_start = event_date - timedelta(days=15)
                pre_end = event_date - timedelta(days=5)
                pre_data = data[pre_start:pre_end]
                
                # Post-event volatility (10 days after)
                post_start = event_date + timedelta(days=5)
                post_end = event_date + timedelta(days=15)
                post_data = data[post_start:post_end]
                
                if len(pre_data) > 0 and len(post_data) > 0:
                    pre_vol = pre_data['Returns'].std() * np.sqrt(252)
                    post_vol = post_data['Returns'].std() * np.sqrt(252)
                    event_vols[event_date] = {'pre': pre_vol, 'post': post_vol}
        
        return {
            'rolling_vol': rolling_vol,
            'event_vols': event_vols
        }
    
    def calculate_iv_rank_percentile(self, symbol, lookback_days=252):
        """Calculate IV rank and percentile"""
        if symbol not in self.data_loader.price_data:
            return None
        
        data = self.data_loader.price_data[symbol]
        current_iv = data['Volatility'].iloc[-1]
        
        # Historical IV over lookback period
        historical_iv = data['Volatility'].tail(lookback_days)
        
        iv_rank = (current_iv - historical_iv.min()) / (historical_iv.max() - historical_iv.min())
        iv_percentile = (historical_iv < current_iv).mean()
        
        return {
            'current_iv': current_iv,
            'iv_rank': iv_rank,
            'iv_percentile': iv_percentile,
            'iv_min': historical_iv.min(),
            'iv_max': historical_iv.max()
        }
    
    def plot_volatility_analysis(self, symbol):
        """Visualize volatility analysis"""
        data = self.data_loader.price_data[symbol]
        vol_data = self.calculate_historical_volatility(symbol)
        iv_stats = self.calculate_iv_rank_percentile(symbol)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price and volatility
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(data.index, data['Close'], 'b-', alpha=0.7, label='Price')
        ax1_twin.plot(data.index, data['Volatility'], 'r-', alpha=0.7, label='Volatility')
        
        ax1.set_ylabel('Price', color='b')
        ax1_twin.set_ylabel('Volatility', color='r')
        ax1.set_title(f'{symbol} - Price and Volatility')
        
        # IV Rank
        ax2 = axes[0, 1]
        if iv_stats:
            ax2.bar(['Min', 'Current', 'Max'], 
                   [iv_stats['iv_min'], iv_stats['current_iv'], iv_stats['iv_max']],
                   color=['green', 'blue', 'red'])
            ax2.set_title(f'IV Range - Rank: {iv_stats["iv_rank"]:.2%}')
        
        # Rolling volatility
        ax3 = axes[1, 0]
        if vol_data and vol_data['rolling_vol'] is not None:
            ax3.plot(data.index, vol_data['rolling_vol'], 'g-')
            ax3.set_title('Rolling 20-Day Volatility')
            ax3.set_ylabel('Volatility')
        
        # Event volatility comparison
        ax4 = axes[1, 1]
        if vol_data and vol_data['event_vols']:
            pre_vols = [v['pre'] for v in vol_data['event_vols'].values()]
            post_vols = [v['post'] for v in vol_data['event_vols'].values()]
            
            x = np.arange(len(pre_vols))
            width = 0.35
            
            ax4.bar(x - width/2, pre_vols, width, label='Pre-Event', alpha=0.7)
            ax4.bar(x + width/2, post_vols, width, label='Post-Event', alpha=0.7)
            ax4.set_title('Event Volatility Comparison')
            ax4.set_ylabel('Volatility')
            ax4.legend()
        
        plt.tight_layout()
        plt.show()


class StrategyConstructor:
    """Module 3: Strategy Constructor for options strategies"""
    
    def __init__(self, data_loader, volatility_model):
        self.data_loader = data_loader
        self.volatility_model = volatility_model
        
    def create_atm_straddle(self, symbol, expiration_days, quantity=1):
        """Create ATM straddle strategy"""
        current_price = self.data_loader.price_data[symbol]['Close'].iloc[-1]
        options = self.data_loader.options_data[symbol]
        
        # Find ATM options
        atm_calls = options[
            (options['Type'] == 'Call') & 
            (options['Strike'] >= current_price * 0.98) & 
            (options['Strike'] <= current_price * 1.02)
        ]
        
        atm_puts = options[
            (options['Type'] == 'Put') & 
            (options['Strike'] >= current_price * 0.98) & 
            (options['Strike'] <= current_price * 1.02)
        ]
        
        if len(atm_calls) == 0 or len(atm_puts) == 0:
            return None
        
        call = atm_calls.iloc[0]
        put = atm_puts.iloc[0]
        
        strategy = {
            'type': 'ATM Straddle',
            'symbol': symbol,
            'expiration_days': expiration_days,
            'legs': [
                {'type': 'Call', 'strike': call['Strike'], 'quantity': quantity, 'price': call['Ask']},
                {'type': 'Put', 'strike': put['Strike'], 'quantity': quantity, 'price': put['Ask']}
            ],
            'total_cost': (call['Ask'] + put['Ask']) * quantity
        }
        
        return strategy
    
    def create_vertical_spread(self, symbol, option_type, short_strike, long_strike, 
                             expiration_days, quantity=1):
        """Create vertical spread strategy"""
        options = self.data_loader.options_data[symbol]
        
        # Find options at specified strikes
        short_leg = options[
            (options['Type'] == option_type) & 
            (options['Strike'] == short_strike)
        ]
        
        long_leg = options[
            (options['Type'] == option_type) & 
            (options['Strike'] == long_strike)
        ]
        
        if len(short_leg) == 0 or len(long_leg) == 0:
            return None
        
        short = short_leg.iloc[0]
        long = long_leg.iloc[0]
        
        if option_type == 'Call':
            # Bull call spread
            net_cost = long['Ask'] - short['Bid']
            strategy_type = 'Bull Call Spread'
        else:
            # Bear put spread
            net_cost = long['Ask'] - short['Bid']
            strategy_type = 'Bear Put Spread'
        
        strategy = {
            'type': strategy_type,
            'symbol': symbol,
            'expiration_days': expiration_days,
            'legs': [
                {'type': option_type, 'strike': short['Strike'], 'quantity': -quantity, 'price': short['Bid']},
                {'type': option_type, 'strike': long['Strike'], 'quantity': quantity, 'price': long['Ask']}
            ],
            'total_cost': net_cost * quantity
        }
        
        return strategy
    
    def create_synthetic_long(self, symbol, strike, expiration_days, quantity=1):
        """Create synthetic long position (long call + short put)"""
        options = self.data_loader.options_data[symbol]
        
        call = options[
            (options['Type'] == 'Call') & 
            (options['Strike'] == strike)
        ]
        
        put = options[
            (options['Type'] == 'Put') & 
            (options['Strike'] == strike)
        ]
        
        if len(call) == 0 or len(put) == 0:
            return None
        
        call = call.iloc[0]
        put = put.iloc[0]
        
        net_cost = call['Ask'] - put['Bid']
        
        strategy = {
            'type': 'Synthetic Long',
            'symbol': symbol,
            'expiration_days': expiration_days,
            'legs': [
                {'type': 'Call', 'strike': call['Strike'], 'quantity': quantity, 'price': call['Ask']},
                {'type': 'Put', 'strike': put['Strike'], 'quantity': -quantity, 'price': put['Bid']}
            ],
            'total_cost': net_cost * quantity
        }
        
        return strategy
    
    def create_strangle(self, symbol, call_strike, put_strike, expiration_days, quantity=1):
        """Create strangle strategy"""
        options = self.data_loader.options_data[symbol]
        
        call = options[
            (options['Type'] == 'Call') & 
            (options['Strike'] == call_strike)
        ]
        
        put = options[
            (options['Type'] == 'Put') & 
            (options['Strike'] == put_strike)
        ]
        
        if len(call) == 0 or len(put) == 0:
            return None
        
        call = call.iloc[0]
        put = put.iloc[0]
        
        total_cost = call['Ask'] + put['Ask']
        
        strategy = {
            'type': 'Strangle',
            'symbol': symbol,
            'expiration_days': expiration_days,
            'legs': [
                {'type': 'Call', 'strike': call['Strike'], 'quantity': quantity, 'price': call['Ask']},
                {'type': 'Put', 'strike': put['Strike'], 'quantity': quantity, 'price': put['Ask']}
            ],
            'total_cost': total_cost * quantity
        }
        
        return strategy


class GreeksEngine:
    """Module 4: Greeks Engine for risk metrics calculation"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def black_scholes_greeks(self, S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes Greeks"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r*T) * norm.cdf(d2))
            vega = S * np.sqrt(T) * norm.pdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r*T) * norm.cdf(-d2))
            vega = S * np.sqrt(T) * norm.pdf(d1)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    def calculate_strategy_greeks(self, strategy, current_price, days_to_expiry, risk_free_rate=0.02):
        """Calculate Greeks for entire strategy"""
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        for leg in strategy['legs']:
            # Estimate IV from option price
            option_price = leg['price']
            strike = leg['strike']
            option_type = leg['type'].lower()
            quantity = leg['quantity']
            
            # Simple IV estimation (in practice, you'd use more sophisticated methods)
            sigma = 0.25  # Default IV estimate
            
            greeks = self.black_scholes_greeks(
                current_price, strike, days_to_expiry/365, risk_free_rate, sigma, option_type
            )
            
            # Scale by quantity
            for greek in total_greeks:
                total_greeks[greek] += greeks[greek] * quantity
        
        return total_greeks
    
    def track_greeks_evolution(self, strategy, price_range, days_to_expiry, risk_free_rate=0.02):
        """Track Greeks evolution across price range"""
        greeks_evolution = []
        
        for price in price_range:
            greeks = self.calculate_strategy_greeks(strategy, price, days_to_expiry, risk_free_rate)
            greeks_evolution.append({
                'price': price,
                **greeks
            })
        
        return pd.DataFrame(greeks_evolution)
    
    def plot_greeks_evolution(self, strategy, price_range, days_to_expiry):
        """Visualize Greeks evolution"""
        greeks_df = self.track_greeks_evolution(strategy, price_range, days_to_expiry)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(greeks_df['price'], greeks_df['delta'])
        axes[0, 0].set_title('Delta')
        axes[0, 0].set_ylabel('Delta')
        
        axes[0, 1].plot(greeks_df['price'], greeks_df['gamma'])
        axes[0, 1].set_title('Gamma')
        axes[0, 1].set_ylabel('Gamma')
        
        axes[1, 0].plot(greeks_df['price'], greeks_df['theta'])
        axes[1, 0].set_title('Theta')
        axes[1, 0].set_ylabel('Theta')
        
        axes[1, 1].plot(greeks_df['price'], greeks_df['vega'])
        axes[1, 1].set_title('Vega')
        axes[1, 1].set_ylabel('Vega')
        
        for ax in axes.flat:
            ax.set_xlabel('Stock Price')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class PnLSimulator:
    """Module 5: PnL & Risk Simulation"""
    
    def __init__(self, data_loader, greeks_engine):
        self.data_loader = data_loader
        self.greeks_engine = greeks_engine
        
    def calculate_option_payoff(self, option_type, strike, current_price, option_price, quantity):
        """Calculate option payoff at expiration"""
        if option_type == 'Call':
            payoff = max(0, current_price - strike) - option_price
        else:  # Put
            payoff = max(0, strike - current_price) - option_price
        
        return payoff * quantity
    
    def calculate_strategy_payoff(self, strategy, final_price):
        """Calculate total strategy payoff"""
        total_payoff = 0
        
        for leg in strategy['legs']:
            leg_payoff = self.calculate_option_payoff(
                leg['type'], leg['strike'], final_price, leg['price'], leg['quantity']
            )
            total_payoff += leg_payoff
        
        return total_payoff
    
    def simulate_pnl_grid(self, strategy, price_range, implied_move=None):
        """Simulate PnL across price grid"""
        pnl_results = []
        
        for price in price_range:
            payoff = self.calculate_strategy_payoff(strategy, price)
            pnl_results.append({
                'price': price,
                'payoff': payoff,
                'return_pct': (payoff / strategy['total_cost']) * 100 if strategy['total_cost'] != 0 else 0
            })
        
        return pd.DataFrame(pnl_results)
    
    def calculate_breakeven_points(self, strategy, price_range):
        """Find breakeven points"""
        pnl_df = self.simulate_pnl_grid(strategy, price_range)
        
        # Find where PnL crosses zero
        zero_crossings = []
        for i in range(len(pnl_df) - 1):
            if (pnl_df['payoff'].iloc[i] <= 0 and pnl_df['payoff'].iloc[i+1] >= 0) or \
               (pnl_df['payoff'].iloc[i] >= 0 and pnl_df['payoff'].iloc[i+1] <= 0):
                # Linear interpolation to find exact breakeven
                x1, y1 = pnl_df['price'].iloc[i], pnl_df['payoff'].iloc[i]
                x2, y2 = pnl_df['price'].iloc[i+1], pnl_df['payoff'].iloc[i+1]
                
                if y2 != y1:
                    breakeven_price = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                    zero_crossings.append(breakeven_price)
        
        return zero_crossings
    
    def calculate_risk_metrics(self, strategy, price_range, implied_move=None):
        """Calculate key risk metrics"""
        pnl_df = self.simulate_pnl_grid(strategy, price_range)
        
        max_profit = pnl_df['payoff'].max()
        max_loss = pnl_df['payoff'].min()
        breakeven_points = self.calculate_breakeven_points(strategy, price_range)
        
        # Expected return (assuming uniform distribution across price range)
        expected_return = pnl_df['payoff'].mean()
        
        # Probability of profit
        prob_profit = (pnl_df['payoff'] > 0).mean()
        
        # Maximum drawdown
        cumulative_pnl = pnl_df['payoff'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max).min()
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'expected_return': expected_return,
            'prob_profit': prob_profit,
            'breakeven_points': breakeven_points,
            'max_drawdown': drawdown
        }
    
    def plot_payoff_diagram(self, strategy, price_range, implied_move=None):
        """Plot strategy payoff diagram"""
        pnl_df = self.simulate_pnl_grid(strategy, price_range)
        
        plt.figure(figsize=(12, 8))
        
        # Main payoff curve
        plt.plot(pnl_df['price'], pnl_df['payoff'], 'b-', linewidth=2, label='Strategy Payoff')
        
        # Zero line
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Implied move lines
        if implied_move:
            current_price = price_range[len(price_range)//2]
            plt.axvline(x=current_price + implied_move, color='r', linestyle='--', alpha=0.7, label=f'Implied Move +{implied_move:.2f}')
            plt.axvline(x=current_price - implied_move, color='r', linestyle='--', alpha=0.7, label=f'Implied Move -{implied_move:.2f}')
        
        # Breakeven points
        breakeven_points = self.calculate_breakeven_points(strategy, price_range)
        for be_point in breakeven_points:
            plt.axvline(x=be_point, color='g', linestyle=':', alpha=0.7, label=f'Breakeven: {be_point:.2f}')
        
        plt.xlabel('Stock Price at Expiration')
        plt.ylabel('Profit/Loss')
        plt.title(f'{strategy["type"]} Payoff Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print risk metrics
        risk_metrics = self.calculate_risk_metrics(strategy, price_range, implied_move)
        print(f"\nRisk Metrics for {strategy['type']}:")
        print(f"Max Profit: ${risk_metrics['max_profit']:.2f}")
        print(f"Max Loss: ${risk_metrics['max_loss']:.2f}")
        print(f"Expected Return: ${risk_metrics['expected_return']:.2f}")
        print(f"Probability of Profit: {risk_metrics['prob_profit']:.1%}")
        print(f"Breakeven Points: {[f'${x:.2f}' for x in risk_metrics['breakeven_points']]}")


class PerformanceAnalyzer:
    """Module 6: Performance & Visualization"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.trades_history = []
        
    def add_trade(self, trade_data):
        """Add completed trade to history"""
        self.trades_history.append(trade_data)
    
    def calculate_trade_performance(self):
        """Calculate overall trade performance"""
        if not self.trades_history:
            return None
        
        df = pd.DataFrame(self.trades_history)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = (df['pnl'] > 0).sum()
        losing_trades = (df['pnl'] < 0).sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(df['pnl'])
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        returns = df['pnl'] / df['initial_cost']
        sharpe_ratio = (returns.mean() - 0.02/252) / returns.std() if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_max_drawdown(self, pnl_series):
        """Calculate maximum drawdown"""
        cumulative = pnl_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max).min()
        return drawdown
    
    def plot_performance_summary(self):
        """Plot comprehensive performance summary"""
        if not self.trades_history:
            print("No trades to analyze")
            return
        
        df = pd.DataFrame(self.trades_history)
        performance = self.calculate_trade_performance()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Cumulative PnL
        cumulative_pnl = df['pnl'].cumsum()
        axes[0, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2)
        axes[0, 0].set_title('Cumulative PnL')
        axes[0, 0].set_ylabel('Cumulative PnL ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win/Loss distribution
        axes[0, 1].hist(df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('PnL Distribution')
        axes[0, 1].set_xlabel('PnL ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Win rate by strategy
        strategy_win_rate = df.groupby('strategy_type')['pnl'].apply(lambda x: (x > 0).mean())
        axes[0, 2].bar(strategy_win_rate.index, strategy_win_rate.values, color='lightgreen')
        axes[0, 2].set_title('Win Rate by Strategy')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Rolling Sharpe ratio
        returns = df['pnl'] / df['initial_cost']
        rolling_sharpe = returns.rolling(window=10).mean() / returns.rolling(window=10).std()
        axes[1, 0].plot(range(len(rolling_sharpe)), rolling_sharpe, 'g-')
        axes[1, 0].set_title('Rolling Sharpe Ratio (10 trades)')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)
        axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance metrics table
        metrics_text = f"""
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Total PnL: ${performance['total_pnl']:.2f}
        Avg Win: ${performance['avg_win']:.2f}
        Avg Loss: ${performance['avg_loss']:.2f}
        Max Drawdown: ${performance['max_drawdown']:.2f}
        Sharpe Ratio: {performance['sharpe_ratio']:.2f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_payoff_heatmap(self, strategy_type, price_range, iv_range):
        """Create payoff heatmap by event size and IV level"""
        heatmap_data = []
        
        for iv in iv_range:
            row = []
            for price in price_range:
                # Simplified payoff calculation for heatmap
                if strategy_type == 'Straddle':
                    payoff = abs(price - 100) - (iv * 10)  # Simplified
                elif strategy_type == 'Strangle':
                    payoff = max(0, abs(price - 100) - 5) - (iv * 8)
                else:
                    payoff = 0
                row.append(payoff)
            heatmap_data.append(row)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=[f'${p:.0f}' for p in price_range[::5]],
                   yticklabels=[f'{iv:.1%}' for iv in iv_range[::2]],
                   cmap='RdYlGn', center=0, annot=False)
        plt.title(f'{strategy_type} Payoff Heatmap')
        plt.xlabel('Stock Price')
        plt.ylabel('Implied Volatility')
        plt.show()


# Main execution and demonstration
def main():
    """Main function to demonstrate the equity derivatives pricing system"""
    print("=== Equity Derivatives Pricing Project ===\n")
    
    # Initialize modules
    data_loader = DataLoader()
    volatility_model = VolatilityModel(data_loader)
    strategy_constructor = StrategyConstructor(data_loader, volatility_model)
    greeks_engine = GreeksEngine(data_loader)
    pnl_simulator = PnLSimulator(data_loader, greeks_engine)
    performance_analyzer = PerformanceAnalyzer(data_loader)
    
    # Load sample data
    print("Loading sample data...")
    symbol = "AAPL"
    data_loader.load_stock_data(symbol, "2023-01-01", "2024-01-01")
    data_loader.load_options_chain(symbol)
    
    # Add sample events
    data_loader.add_event(symbol, datetime(2023, 7, 25), "Earnings", "Q3 2023 Earnings")
    data_loader.add_event(symbol, datetime(2023, 10, 26), "Earnings", "Q4 2023 Earnings")
    
    # Volatility analysis
    print("\n=== Volatility Analysis ===")
    iv_stats = volatility_model.calculate_iv_rank_percentile(symbol)
    if iv_stats:
        print(f"Current IV: {iv_stats['current_iv']:.2%}")
        print(f"IV Rank: {iv_stats['iv_rank']:.2%}")
        print(f"IV Percentile: {iv_stats['iv_percentile']:.2%}")
    
    # Create strategies
    print("\n=== Strategy Construction ===")
    current_price = data_loader.price_data[symbol]['Close'].iloc[-1]
    
    # ATM Straddle
    straddle = strategy_constructor.create_atm_straddle(symbol, 30)
    if straddle:
        print(f"Created {straddle['type']} - Cost: ${straddle['total_cost']:.2f}")
    
    # Vertical Spread
    call_spread = strategy_constructor.create_vertical_spread(
        symbol, "Call", current_price, current_price + 10, 30
    )
    if call_spread:
        print(f"Created {call_spread['type']} - Cost: ${call_spread['total_cost']:.2f}")
    
    # Greeks analysis
    print("\n=== Greeks Analysis ===")
    if straddle:
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
        greeks_engine.plot_greeks_evolution(straddle, price_range, 30)
    
    # PnL simulation
    print("\n=== PnL Simulation ===")
    if straddle:
        pnl_simulator.plot_payoff_diagram(straddle, price_range, implied_move=current_price * 0.05)
    
    # Performance analysis
    print("\n=== Performance Analysis ===")
    # Add sample trades
    sample_trades = [
        {'strategy_type': 'Straddle', 'pnl': 150, 'initial_cost': 500},
        {'strategy_type': 'Call Spread', 'pnl': -50, 'initial_cost': 200},
        {'strategy_type': 'Strangle', 'pnl': 300, 'initial_cost': 400},
        {'strategy_type': 'Straddle', 'pnl': -100, 'initial_cost': 600},
        {'strategy_type': 'Put Spread', 'pnl': 75, 'initial_cost': 150}
    ]
    
    for trade in sample_trades:
        performance_analyzer.add_trade(trade)
    
    performance_analyzer.plot_performance_summary()
    
    print("\n=== Project Complete ===")
    print("This project demonstrates:")
    print("✓ Data loading and management")
    print("✓ Volatility modeling and analysis")
    print("✓ Options strategy construction")
    print("✓ Greeks calculation and risk management")
    print("✓ PnL simulation and payoff analysis")
    print("✓ Performance tracking and visualization")


if __name__ == "__main__":
    main()
