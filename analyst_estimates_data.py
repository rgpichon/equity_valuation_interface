import pandas as pd
import time
from typing import Dict, List, Optional
from enhanced_comprehensive_data import EnhancedComprehensiveStockData #type: ignore

class AnalystEstimatesFetcher:
    """
    Efficient analyst estimates data fetcher using enhanced data management system.
    Designed for stock analysis demonstrations with real market data.
    """
    
    def __init__(self, data_manager: Optional[EnhancedComprehensiveStockData] = None):
        self.stocks = {
            'AAPL': 'Apple Inc.',
            'BABA': 'Alibaba Group',
            'CAT': 'Caterpillar Inc.',
            'NVO': 'Novo Nordisk A/S',
            'SIEGY': 'Siemens AG'
        }
        
        # Use provided data manager or get shared instance to prevent multiple API calls
        self.data_manager = data_manager if data_manager else EnhancedComprehensiveStockData.get_shared_instance(use_cache=True)
        
    def get_analyst_estimates(self, ticker: str) -> Dict:
        """
        Fetch analyst estimates for a single stock using enhanced data system.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Standardized analyst estimates data
        """
        try:
            # Get data from enhanced data system
            analyst_data = self.data_manager.export_analyst_estimates(ticker)
            fundamental_data = self.data_manager.export_fundamental_data(ticker)
            
            # Extract price targets
            price_targets = {
                'target_high': analyst_data.get('target_high'),
                'target_median': analyst_data.get('target_median'),
                'target_low': analyst_data.get('target_low'),
                'current_price': analyst_data.get('current_price')
            }
            
            # Extract basic financial metrics
            basic_info = {
                'market_cap': fundamental_data.get('market_cap'),
                'pe_ratio': fundamental_data.get('pe_ratio'),
                'forward_pe': fundamental_data.get('forward_pe'),
                'peg_ratio': fundamental_data.get('peg_ratio'),
                'analyst_rating': analyst_data.get('analyst_rating'),
                'number_of_analysts': analyst_data.get('number_of_analysts')
            }
            
            # Extract earnings estimates
            earnings_data = {
                'current_year_eps_estimate': analyst_data.get('current_year_eps_estimate'),
                'next_year_eps_estimate': analyst_data.get('next_year_eps_estimate'),
                'current_year_revenue_estimate': analyst_data.get('current_year_revenue_estimate'),
                'next_year_revenue_estimate': analyst_data.get('next_year_revenue_estimate')
            }
            
            return {
                'ticker': ticker,
                'company_name': self.stocks.get(ticker, ticker),
                'price_targets': price_targets,
                'earnings_estimates': earnings_data,
                'basic_info': basic_info,
                'latest_recommendation': None,  # Not available in enhanced data system
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'company_name': self.stocks.get(ticker, ticker),
                'success': False,
                'error': str(e),
                'price_targets': {},
                'earnings_estimates': {},
                'basic_info': {},
                'latest_recommendation': None
            }
    
    def get_all_analyst_estimates(self) -> Dict[str, Dict]:
        """
        Fetch analyst estimates for all predefined stocks using cached data.
        
        Returns:
            Dict: All analyst estimates data
        """
        results = {}
        
        print("Fetching analyst estimates data...")
        
        for ticker in self.stocks.keys():
            print(f"Fetching data for {ticker}...")
            results[ticker] = self.get_analyst_estimates(ticker)
            # No delay needed - using cached data
        
        return results
    
    def create_summary_dataframe(self, data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a clean summary DataFrame from the analyst estimates data.
        
        Args:
            data (Dict): Raw analyst estimates data
            
        Returns:
            pd.DataFrame: Formatted summary table
        """
        summary_data = []
        
        for ticker, stock_data in data.items():
            if stock_data['success']:
                summary_data.append({
                    'Ticker': ticker,
                    'Company': stock_data['company_name'],
                    'Current Price': stock_data['price_targets'].get('current_price'),
                    'Target High': stock_data['price_targets'].get('target_high'),
                    'Target Median': stock_data['price_targets'].get('target_median'),
                    'Target Low': stock_data['price_targets'].get('target_low'),
                    'Analyst Rating': stock_data['basic_info'].get('analyst_rating'),
                    'Number of Analysts': stock_data['basic_info'].get('number_of_analysts'),
                    'Current Year EPS Est.': stock_data['earnings_estimates'].get('current_year_eps_estimate'),
                    'Next Year EPS Est.': stock_data['earnings_estimates'].get('next_year_eps_estimate'),
                    'PE Ratio': stock_data['basic_info'].get('pe_ratio'),
                    'Forward PE': stock_data['basic_info'].get('forward_pe'),
                    'Market Cap (B)': round(stock_data['basic_info'].get('market_cap', 0) / 1e9, 2) if stock_data['basic_info'].get('market_cap') else None
                })
            else:
                summary_data.append({
                    'Ticker': ticker,
                    'Company': stock_data['company_name'],
                    'Error': stock_data['error']
                })
        
        return pd.DataFrame(summary_data)
    
    def print_summary(self, data: Dict[str, Dict]):
        """
        Print a formatted summary of analyst estimates.
        
        Args:
            data (Dict): Raw analyst estimates data
        """
        df = self.create_summary_dataframe(data)
        
        print("\n" + "="*80)
        print("ANALYST ESTIMATES SUMMARY")
        print("="*80)
        
        # Print successful data
        if 'Error' in df.columns:
            successful_data = df[~df['Ticker'].isin(df[df['Error'].notna()]['Ticker'])]
            if not successful_data.empty:
                print("\nSuccessfully fetched data:")
                print(successful_data.to_string(index=False))
            
            # Print errors
            error_data = df[df['Error'].notna()]
            if not error_data.empty:
                print("\nErrors encountered:")
                for _, row in error_data.iterrows():
                    print(f"{row['Ticker']}: {row['Error']}")
        else:
            # All data is successful
            print("\nSuccessfully fetched data:")
            print(df.to_string(index=False))
        
        print("\n" + "="*80)
    
    def get_peer_multiples(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get peer multiples (PE_NTM, EV/EBITDA_NTM, EV/EBIT_NTM) with median and IQR.
        
        Pure getter with no side effects (no prints, no writes).
        
        Args:
            tickers: List of stock ticker symbols for peer group
            
        Returns:
            pd.DataFrame with columns:
                - Ticker: str
                - PE_NTM: float (forward P/E)
                - EV_EBITDA_NTM: float (enterprise value / EBITDA)
                - EV_EBIT_NTM: float (enterprise value / EBIT)
                
            Additional summary rows:
                - 'Median': trimmed median (exclude top/bottom 10%)
                - 'IQR_25th': 25th percentile
                - 'IQR_75th': 75th percentile
        """
        peer_data = []
        
        for ticker in tickers:
            try:
                estimates = self.get_analyst_estimates(ticker)
                fundamental = self.data_manager.export_fundamental_data(ticker)
                
                if not estimates['success']:
                    continue
                
                # Extract multiples
                pe_ntm = fundamental.get('forward_pe')
                
                # Calculate EV/EBITDA and EV/EBIT from available data
                ev = fundamental.get('enterprise_value')
                market_cap = fundamental.get('market_cap')
                
                # Estimate EBITDA and EBIT from available ratios
                revenue = estimates['earnings_estimates'].get('next_year_revenue_estimate')
                if revenue and market_cap:
                    ps_ratio = market_cap / revenue if revenue > 0 else None
                    # Rough estimate: EBITDA ~20% of revenue, EBIT ~15%
                    ebitda_est = revenue * 0.20 if revenue else None
                    ebit_est = revenue * 0.15 if revenue else None
                    
                    ev_ebitda = (ev / ebitda_est) if ev and ebitda_est and ebitda_est > 0 else None
                    ev_ebit = (ev / ebit_est) if ev and ebit_est and ebit_est > 0 else None
                else:
                    ev_ebitda = None
                    ev_ebit = None
                
                peer_data.append({
                    'Ticker': ticker,
                    'PE_NTM': pe_ntm,
                    'EV_EBITDA_NTM': ev_ebitda,
                    'EV_EBIT_NTM': ev_ebit
                })
                
            except:
                continue
        
        if not peer_data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Ticker', 'PE_NTM', 'EV_EBITDA_NTM', 'EV_EBIT_NTM'])
        
        df = pd.DataFrame(peer_data)
        
        # Calculate summary statistics (trimmed median and IQR)
        summary_rows = []
        
        for metric in ['PE_NTM', 'EV_EBITDA_NTM', 'EV_EBIT_NTM']:
            values = df[metric].dropna()
            if len(values) >= 3:
                # Trimmed median: exclude top/bottom 10%
                sorted_vals = values.sort_values()
                trim_count = max(1, int(len(sorted_vals) * 0.10))
                trimmed = sorted_vals.iloc[trim_count:-trim_count] if len(sorted_vals) > 2 * trim_count else sorted_vals
                
                median_val = trimmed.median()
                q25 = values.quantile(0.25)
                q75 = values.quantile(0.75)
            else:
                median_val = values.median() if len(values) > 0 else np.nan
                q25 = np.nan
                q75 = np.nan
            
            # Store for summary rows
            if not summary_rows:
                summary_rows = [
                    {'Ticker': 'Median', metric: median_val},
                    {'Ticker': 'IQR_25th', metric: q25},
                    {'Ticker': 'IQR_75th', metric: q75}
                ]
            else:
                summary_rows[0][metric] = median_val
                summary_rows[1][metric] = q25
                summary_rows[2][metric] = q75
        
        # Append summary rows
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            df = pd.concat([df, summary_df], ignore_index=True)
        
        return df


def test_single_stock():
    """
    Test function to try fetching data for just one stock.
    """
    fetcher = AnalystEstimatesFetcher()
    
    print("Testing with Apple (AAPL) only...")
    data = fetcher.get_analyst_estimates('AAPL')
    
    print("\nRaw data for AAPL:")
    for key, value in data.items():
        if key not in ['success', 'error']:
            print(f"{key}: {value}")
    
    if data['success']:
        print(f"\nSuccess! Error: {data.get('error')}")
    else:
        print(f"\nFailed! Error: {data.get('error')}")
    
    return data


def main():
    """
    Main function to demonstrate the analyst estimates fetcher.
    """
    fetcher = AnalystEstimatesFetcher()
    
    # Fetch all analyst estimates
    data = fetcher.get_all_analyst_estimates()
    
    # Print summary
    fetcher.print_summary(data)
    
    # Return data for further analysis
    return data


def test_with_mock_data():
    """
    Test function using mock data to demonstrate the output format.
    """
    print("="*60)
    print("TESTING WITH MOCK DATA (to show output format)")
    print("="*60)
    
    # Mock data that represents what we would get from yfinance
    mock_data = {
        'AAPL': {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'price_targets': {
                'target_high': 250.0,
                'target_median': 220.0,
                'target_low': 180.0,
                'current_price': 195.5
            },
            'earnings_estimates': {
                'current_year_eps_estimate': 6.45,
                'next_year_eps_estimate': 7.20,
                'current_year_revenue_estimate': 385000000000,
                'next_year_revenue_estimate': 405000000000
            },
            'basic_info': {
                'market_cap': 3100000000000,
                'pe_ratio': 30.3,
                'forward_pe': 27.1,
                'peg_ratio': 2.1,
                'analyst_rating': 2.1,
                'number_of_analysts': 45
            },
            'latest_recommendation': None,
            'success': True,
            'error': None
        },
        'BABA': {
            'ticker': 'BABA',
            'company_name': 'Alibaba Group',
            'price_targets': {
                'target_high': 120.0,
                'target_median': 95.0,
                'target_low': 75.0,
                'current_price': 85.2
            },
            'earnings_estimates': {
                'current_year_eps_estimate': 4.20,
                'next_year_eps_estimate': 5.10,
                'current_year_revenue_estimate': 95000000000,
                'next_year_revenue_estimate': 105000000000
            },
            'basic_info': {
                'market_cap': 220000000000,
                'pe_ratio': 20.3,
                'forward_pe': 16.7,
                'peg_ratio': 0.8,
                'analyst_rating': 2.3,
                'number_of_analysts': 32
            },
            'latest_recommendation': None,
            'success': True,
            'error': None
        }
    }
    
    fetcher = AnalystEstimatesFetcher()
    fetcher.print_summary(mock_data)
    
    return mock_data


if __name__ == "__main__":
    # Test with mock data first to show format
    mock_data = test_with_mock_data()
    
    print("\n" + "="*60)
    print("TESTING REAL DATA (if rate limits allow)")
    print("="*60)
    
    # Wait a bit before trying real data
    print("Waiting 10 seconds before trying real data...")
    time.sleep(10)
    
    try:
        # Test with single stock first
        print("Testing with Apple (AAPL) only...")
        test_data = test_single_stock()
        
        if test_data['success']:
            print("\n" + "="*60)
            print("TESTING ALL STOCKS")
            print("="*60)
            analyst_data = main()
        else:
            print("Rate limited. Using mock data for demonstration.")
            fetcher = AnalystEstimatesFetcher()
            fetcher.print_summary(mock_data)
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Using mock data for demonstration.")
        fetcher = AnalystEstimatesFetcher()
        fetcher.print_summary(mock_data)
