import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from dataclasses import dataclass
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for different data sources"""
    name: str
    rate_limit_per_minute: int
    rate_limit_per_day: int
    priority: int  # Lower number = higher priority
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class RateLimiter:
    """Simple rate limiter to manage API calls"""
    
    def __init__(self, calls_per_minute: int = 10, calls_per_day: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls = []
        self.daily_calls = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def can_make_call(self) -> bool:
        """Check if we can make an API call without exceeding limits"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_calls.clear()
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Clean old minute calls
        minute_ago = now - timedelta(minutes=1)
        self.minute_calls = [call_time for call_time in self.minute_calls if call_time > minute_ago]
        
        # Check limits
        if len(self.minute_calls) >= self.calls_per_minute:
            return False
        if len(self.daily_calls) >= self.calls_per_day:
            return False
        
        return True
    
    def record_call(self):
        """Record that an API call was made"""
        now = datetime.now()
        self.minute_calls.append(now)
        self.daily_calls.append(now)
    
    def wait_time(self) -> float:
        """Calculate how long to wait before next call is allowed"""
        if not self.minute_calls:
            return 0
        
        oldest_minute_call = min(self.minute_calls)
        minute_threshold = oldest_minute_call + timedelta(minutes=1)
        wait_seconds = (minute_threshold - datetime.now()).total_seconds()
        
        return max(0, wait_seconds)

class DataCache:
    """SQLite-based cache for storing and retrieving financial data"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT,
                last_updated TEXT,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        # Fundamental data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_data (
                ticker TEXT PRIMARY KEY,
                data TEXT,
                source TEXT,
                last_updated TEXT
            )
        ''')
        
        # Analyst estimates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                ticker TEXT PRIMARY KEY,
                data TEXT,
                source TEXT,
                last_updated TEXT
            )
        ''')
        
        # Technical indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                ticker TEXT,
                date TEXT,
                data TEXT,
                source TEXT,
                last_updated TEXT,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def is_data_fresh(self, ticker: str, data_type: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is still fresh"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_type == 'ohlcv':
            cursor.execute('''
                SELECT MAX(last_updated) FROM ohlcv_data 
                WHERE ticker = ? AND date >= date('now', '-1 day')
            ''', (ticker,))
        else:
            table_name = 'analyst_estimates' if data_type == 'analyst' else f'{data_type}_data'
            cursor.execute(f'''
                SELECT last_updated FROM {table_name} WHERE ticker = ?
            ''', (ticker,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or not result[0]:
            return False
        
        last_updated = datetime.fromisoformat(result[0])
        max_age = timedelta(hours=max_age_hours)
        
        return datetime.now() - last_updated < max_age
    
    def store_ohlcv_data(self, ticker: str, df: pd.DataFrame, source: str):
        """Store OHLCV data in cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (ticker, date, open, high, low, close, volume, source, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker, row['Date'].isoformat(), row['Open'], row['High'], 
                row['Low'], row['Close'], row['Volume'], source, 
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def get_ohlcv_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """Retrieve OHLCV data from cache"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT date, open, high, low, close, volume 
            FROM ohlcv_data 
            WHERE ticker = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(ticker, days))
        conn.close()
        
        if df.empty:
            return None
        
        # Standardize column names to uppercase first
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        return df
    
    def store_fundamental_data(self, ticker: str, data: Dict, source: str):
        """Store fundamental data in cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO fundamental_data 
            (ticker, data, source, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (ticker, json.dumps(data), source, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Retrieve fundamental data from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT data FROM fundamental_data WHERE ticker = ?', (ticker,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def store_analyst_estimates(self, ticker: str, data: Dict, source: str):
        """Store analyst estimates in cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO analyst_estimates 
            (ticker, data, source, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (ticker, json.dumps(data), source, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_analyst_estimates(self, ticker: str) -> Optional[Dict]:
        """Retrieve analyst estimates from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT data FROM analyst_estimates WHERE ticker = ?', (ticker,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None

class MultiSourceDataFetcher:
    """Fetches data from multiple sources with intelligent fallbacks"""
    
    def __init__(self):
        self.cache = DataCache()
        
        # Configure API sources with your actual keys
        self.apis = {
            'yfinance': APIConfig(
                name='yfinance',
                rate_limit_per_minute=10,
                rate_limit_per_day=500,
                priority=1
            ),
            'fmp': APIConfig(
                name='fmp',
                rate_limit_per_minute=5,
                rate_limit_per_day=250,
                priority=2,
                api_key=os.getenv('FMP_API_KEY'),  # Get from environment variable
                base_url='https://financialmodelingprep.com/api/v3'
            ),
            'polygon': APIConfig(
                name='polygon',
                rate_limit_per_minute=5,
                rate_limit_per_day=100,
                priority=3,
                api_key=os.getenv('POLYGON_API_KEY'),  # Get from environment variable
                base_url='https://api.polygon.io/v2'
            )
        }
        
        # Initialize rate limiters
        self.rate_limiters = {}
        for api_name, config in self.apis.items():
            self.rate_limiters[api_name] = RateLimiter(
                calls_per_minute=config.rate_limit_per_minute,
                calls_per_day=config.rate_limit_per_day
            )
    
    def _wait_for_rate_limit(self, api_name: str):
        """Wait if rate limit is exceeded"""
        rate_limiter = self.rate_limiters[api_name]
        
        while not rate_limiter.can_make_call():
            wait_time = rate_limiter.wait_time()
            if wait_time > 0:
                logger.info(f"Rate limit reached for {api_name}. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        rate_limiter.record_call()
    
    def fetch_ohlcv_yfinance(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from yfinance"""
        try:
            self._wait_for_rate_limit('yfinance')
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
            # Standardize column names
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Store in cache
            self.cache.store_ohlcv_data(ticker, data, 'yfinance')
            
            logger.info(f"✓ Fetched OHLCV data for {ticker} from yfinance")
            return data
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch OHLCV data for {ticker} from yfinance: {e}")
            return None
    
    def fetch_fundamental_yfinance(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from yfinance including financial statements"""
        try:
            self._wait_for_rate_limit('yfinance')
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) < 5:
                return None
            
            # Extract key fundamental metrics (ratios)
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta')
            }
            
            # Add actual financial statement data (NEW!)
            # These are absolute values, not ratios
            fundamental_data.update({
                'total_revenue': info.get('totalRevenue'),
                'ebitda': info.get('ebitda'),
                'ebit': info.get('ebit'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                'shares_outstanding': info.get('sharesOutstanding')
            })
            
            # Try to get more detailed data from financial statements
            try:
                # Get latest financials from income statement
                financials = stock.financials
                if not financials.empty and len(financials.columns) > 0:
                    latest_year = financials.columns[0]
                    if 'Total Revenue' in financials.index:
                        fundamental_data['total_revenue'] = fundamental_data.get('total_revenue') or financials.loc['Total Revenue', latest_year]
                    if 'EBITDA' in financials.index:
                        fundamental_data['ebitda'] = fundamental_data.get('ebitda') or financials.loc['EBITDA', latest_year]
                    if 'EBIT' in financials.index:
                        fundamental_data['ebit'] = fundamental_data.get('ebit') or financials.loc['EBIT', latest_year]
            except Exception as e:
                logger.debug(f"Could not fetch detailed financials for {ticker}: {e}")
            
            # Try to get balance sheet data
            try:
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                    latest_year = balance_sheet.columns[0]
                    if 'Total Debt' in balance_sheet.index:
                        fundamental_data['total_debt'] = fundamental_data.get('total_debt') or balance_sheet.loc['Total Debt', latest_year]
                    if 'Cash And Cash Equivalents' in balance_sheet.index:
                        fundamental_data['total_cash'] = fundamental_data.get('total_cash') or balance_sheet.loc['Cash And Cash Equivalents', latest_year]
            except Exception as e:
                logger.debug(f"Could not fetch balance sheet for {ticker}: {e}")
            
            # Try to get cash flow data
            try:
                cashflow = stock.cashflow
                if not cashflow.empty and len(cashflow.columns) > 0:
                    latest_year = cashflow.columns[0]
                    if 'Free Cash Flow' in cashflow.index:
                        fundamental_data['free_cash_flow'] = fundamental_data.get('free_cash_flow') or cashflow.loc['Free Cash Flow', latest_year]
                    if 'Operating Cash Flow' in cashflow.index:
                        fundamental_data['operating_cash_flow'] = fundamental_data.get('operating_cash_flow') or cashflow.loc['Operating Cash Flow', latest_year]
                    if 'Capital Expenditures' in cashflow.index:
                        fundamental_data['capital_expenditures'] = abs(cashflow.loc['Capital Expenditures', latest_year])
            except Exception as e:
                logger.debug(f"Could not fetch cash flow for {ticker}: {e}")
            
            # Store raw info for additional access if needed
            fundamental_data['raw_info'] = info
            
            # Store in cache
            self.cache.store_fundamental_data(ticker, fundamental_data, 'yfinance')
            
            logger.info(f"✓ Fetched fundamental data for {ticker} from yfinance")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch fundamental data for {ticker} from yfinance: {e}")
            return None
    
    def fetch_analyst_estimates_yfinance(self, ticker: str) -> Optional[Dict]:
        """Fetch analyst estimates from yfinance"""
        try:
            self._wait_for_rate_limit('yfinance')
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return None
            
            analyst_data = {
                'current_price': info.get('currentPrice'),
                'target_high': info.get('targetHighPrice'),
                'target_median': info.get('targetMedianPrice'),
                'target_low': info.get('targetLowPrice'),
                'analyst_rating': info.get('recommendationMean'),
                'number_of_analysts': info.get('numberOfAnalystOpinions'),
                'current_year_eps_estimate': info.get('currentYearEpsEstimate'),
                'next_year_eps_estimate': info.get('nextYearEpsEstimate'),
                'current_year_revenue_estimate': info.get('currentYearRevenueEstimate'),
                'next_year_revenue_estimate': info.get('nextYearRevenueEstimate')
            }
            
            # Store in cache
            self.cache.store_analyst_estimates(ticker, analyst_data, 'yfinance')
            
            logger.info(f"✓ Fetched analyst estimates for {ticker} from yfinance")
            return analyst_data
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch analyst estimates for {ticker} from yfinance: {e}")
            return None
    
    def fetch_ohlcv_fmp(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Financial Modeling Prep"""
        try:
            if not self.apis['fmp'].api_key:
                logger.warning("FMP API key not configured")
                return None
            
            self._wait_for_rate_limit('fmp')
            
            url = f"{self.apis['fmp'].base_url}/historical-price-full/{ticker}"
            params = {'apikey': self.apis['fmp'].api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'historical' not in data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Standardize column names
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Store in cache
            self.cache.store_ohlcv_data(ticker, df, 'fmp')
            
            logger.info(f"✓ Fetched OHLCV data for {ticker} from FMP")
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch OHLCV data for {ticker} from FMP: {e}")
            return None
    
    def fetch_fundamental_fmp(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from Financial Modeling Prep"""
        try:
            if not self.apis['fmp'].api_key:
                logger.warning("FMP API key not configured")
                return None
            
            self._wait_for_rate_limit('fmp')
            
            # Fetch key metrics
            url = f"{self.apis['fmp'].base_url}/key-metrics/{ticker}"
            params = {'apikey': self.apis['fmp'].api_key, 'limit': 1}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or len(data) == 0:
                return None
            
            metrics = data[0]
            
            # Extract and standardize metrics
            fundamental_data = {
                'market_cap': metrics.get('marketCapitalization'),
                'enterprise_value': metrics.get('enterpriseValue'),
                'pe_ratio': metrics.get('peRatio'),
                'forward_pe': metrics.get('priceEarningsToGrowthRatio'),
                'peg_ratio': metrics.get('pegRatio'),
                'pb_ratio': metrics.get('priceToBookRatio'),
                'ps_ratio': metrics.get('priceToSalesRatio'),
                'debt_to_equity': metrics.get('debtToEquity'),
                'current_ratio': metrics.get('currentRatio'),
                'roe': metrics.get('returnOnEquity'),
                'roa': metrics.get('returnOnAssets'),
                'revenue_growth': metrics.get('revenueGrowth'),
                'earnings_growth': metrics.get('earningsGrowth'),
                'dividend_yield': metrics.get('dividendYield'),
                'beta': metrics.get('beta')
            }
            
            # Store in cache
            self.cache.store_fundamental_data(ticker, fundamental_data, 'fmp')
            
            logger.info(f"✓ Fetched fundamental data for {ticker} from FMP")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch fundamental data for {ticker} from FMP: {e}")
            return None
    
    def fetch_ohlcv_polygon(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Polygon.io"""
        try:
            if not self.apis['polygon'].api_key:
                logger.warning("Polygon API key not configured")
                return None
            
            self._wait_for_rate_limit('polygon')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.apis['polygon'].base_url}/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apikey': self.apis['polygon'].api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return None
            
            # Convert to DataFrame
            results = data['results']
            df_data = []
            
            for item in results:
                df_data.append({
                    'Date': pd.to_datetime(item['t'], unit='ms'),
                    'Open': item['o'],
                    'High': item['h'],
                    'Low': item['l'],
                    'Close': item['c'],
                    'Volume': item['v']
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('Date')
            
            # Store in cache
            self.cache.store_ohlcv_data(ticker, df, 'polygon')
            
            logger.info(f"✓ Fetched OHLCV data for {ticker} from Polygon")
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch OHLCV data for {ticker} from Polygon: {e}")
            return None

class EnhancedComprehensiveStockData:
    """
    Enhanced centralized data management system for stock analysis.
    Uses multiple data sources with intelligent caching and rate limiting.
    
    CRITICAL FEATURE: Daily loading mechanism prevents multiple API calls.
    Data is fetched only once per day to avoid rate limits and stale data issues.
    """
    
    # Class variables for daily loading control
    _daily_loaded = False
    _daily_load_date = None
    _shared_instance = None
    
    def __init__(self, use_cache: bool = True, max_age_hours: int = 24):
        self.stocks = {
            'AAPL': 'Apple Inc.',
            'BABA': 'Alibaba Group Holding Limited',
            'CAT': 'Caterpillar Inc.',
            'NVO': 'Novo Nordisk A/S',
            'SIEGY': 'Siemens AG'
        }
        
        self.use_cache = use_cache
        self.max_age_hours = max_age_hours
        self.data_fetcher = MultiSourceDataFetcher()
        
        # Initialize data storage
        self.ohlcv_data = {}
        self.analyst_estimates = {}
        self.fundamental_data = {}
        self.technical_indicators = {}
        
        # Check if we need to load data today
        self._check_and_load_daily_data()
    
    @classmethod
    def get_shared_instance(cls, use_cache: bool = True, max_age_hours: int = 24):
        """Get shared instance to prevent multiple data loading"""
        if cls._shared_instance is None:
            cls._shared_instance = cls(use_cache, max_age_hours)
        return cls._shared_instance
    
    @classmethod
    def reset_daily_loading(cls):
        """Reset daily loading status (for testing or manual refresh)"""
        cls._daily_loaded = False
        cls._daily_load_date = None
        cls._shared_instance = None
        logger.info("🔄 Daily loading status reset - next instance will fetch fresh data")
    
    def _check_and_load_daily_data(self):
        """Check if data needs to be loaded today and load if necessary"""
        from datetime import datetime, date
        
        today = date.today()
        
        # Check if we already loaded data today
        if self._daily_loaded and self._daily_load_date == today:
            logger.info("✓ Data already loaded today - using cached data")
            self._load_from_cache_only()
            return
        
        # Check if we have fresh cached data (less than max_age_hours old)
        if self.use_cache and self._has_fresh_cached_data():
            logger.info("✓ Fresh cached data available - using cache")
            self._load_from_cache_only()
            return
        
        # Load fresh data from APIs
        logger.info("🔄 Loading fresh data from APIs (first load today)")
        self._load_all_data()
        
        # Mark as loaded today
        self._daily_loaded = True
        self._daily_load_date = today
    
    def _has_fresh_cached_data(self) -> bool:
        """Check if we have fresh cached data for all stocks"""
        try:
            for ticker in self.stocks.keys():
                if not self.data_fetcher.cache.is_data_fresh(ticker, 'ohlcv', self.max_age_hours):
                    return False
                if not self.data_fetcher.cache.is_data_fresh(ticker, 'fundamental', self.max_age_hours):
                    return False
                if not self.data_fetcher.cache.is_data_fresh(ticker, 'analyst', self.max_age_hours):
                    return False
            return True
        except:
            return False
    
    def _load_from_cache_only(self):
        """Load data from cache only (no API calls)"""
        logger.info("📁 Loading data from cache only...")
        
        for ticker in self.stocks.keys():
            try:
                # Load OHLCV data
                ohlcv_data = self.data_fetcher.cache.get_ohlcv_data(ticker)
                if ohlcv_data is not None and not ohlcv_data.empty:
                    self.ohlcv_data[ticker] = ohlcv_data
                    logger.info(f"✓ Loaded cached OHLCV data for {ticker}")
                else:
                    logger.warning(f"⚠ No cached OHLCV data for {ticker}")
                
                # Load fundamental data
                fundamental_data = self.data_fetcher.cache.get_fundamental_data(ticker)
                if fundamental_data is not None:
                    self.fundamental_data[ticker] = fundamental_data
                    logger.info(f"✓ Loaded cached fundamental data for {ticker}")
                else:
                    logger.warning(f"⚠ No cached fundamental data for {ticker}")
                
                # Load analyst estimates
                analyst_data = self.data_fetcher.cache.get_analyst_estimates(ticker)
                if analyst_data is not None:
                    self.analyst_estimates[ticker] = analyst_data
                    logger.info(f"✓ Loaded cached analyst data for {ticker}")
                else:
                    logger.warning(f"⚠ No cached analyst data for {ticker}")
                
                # Generate technical indicators only if we have OHLCV data
                if ticker in self.ohlcv_data and not self.ohlcv_data[ticker].empty:
                    self._generate_technical_indicators(ticker)
                
            except Exception as e:
                logger.warning(f"⚠ Could not load cached data for {ticker}: {e}")
    
    def _load_all_data(self):
        """Load data for all stocks with intelligent caching"""
        logger.info("🌐 Loading fresh data from APIs...")
        
        for ticker in self.stocks.keys():
            logger.info(f"Loading data for {ticker}...")
            self._load_stock_data(ticker)
            
            # Add delay between stocks to avoid rate limiting
            time.sleep(2)
    
    def _load_stock_data(self, ticker: str):
        """Load data for a single stock with caching and fallbacks"""
        
        # Try to get fresh cached data first
        if self.use_cache and self._try_load_from_cache(ticker):
            return
        
        # If cache miss or stale, fetch fresh data
        self._fetch_fresh_data(ticker)
    
    def _try_load_from_cache(self, ticker: str) -> bool:
        """Try to load fresh data from cache"""
        try:
            # Check OHLCV data
            if self.data_fetcher.cache.is_data_fresh(ticker, 'ohlcv', self.max_age_hours):
                ohlcv_data = self.data_fetcher.cache.get_ohlcv_data(ticker)
                if ohlcv_data is not None:
                    self.ohlcv_data[ticker] = ohlcv_data
                    logger.info(f"✓ Loaded fresh OHLCV data for {ticker} from cache")
                else:
                    return False
            else:
                return False
            
            # Check fundamental data
            if self.data_fetcher.cache.is_data_fresh(ticker, 'fundamental', self.max_age_hours):
                fundamental_data = self.data_fetcher.cache.get_fundamental_data(ticker)
                if fundamental_data is not None:
                    self.fundamental_data[ticker] = fundamental_data
                    logger.info(f"✓ Loaded fresh fundamental data for {ticker} from cache")
                else:
                    return False
            else:
                return False
            
            # Check analyst estimates
            if self.data_fetcher.cache.is_data_fresh(ticker, 'analyst', self.max_age_hours):
                analyst_data = self.data_fetcher.cache.get_analyst_estimates(ticker)
                if analyst_data is not None:
                    self.analyst_estimates[ticker] = analyst_data
                    logger.info(f"✓ Loaded fresh analyst estimates for {ticker} from cache")
                else:
                    return False
            else:
                return False
            
            # Generate technical indicators
            self._generate_technical_indicators(ticker)
            
            logger.info(f"✓ All fresh data loaded for {ticker} from cache")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading cached data for {ticker}: {e}")
            return False
    
    def _fetch_fresh_data(self, ticker: str):
        """Fetch fresh data from APIs with fallback strategy"""
        
        # Try yfinance first (highest priority)
        ohlcv_data = self.data_fetcher.fetch_ohlcv_yfinance(ticker)
        fundamental_data = self.data_fetcher.fetch_fundamental_yfinance(ticker)
        analyst_data = self.data_fetcher.fetch_analyst_estimates_yfinance(ticker)
        
        # If yfinance fails, try FMP
        if ohlcv_data is None:
            logger.warning(f"yfinance failed for {ticker}, trying FMP...")
            ohlcv_data = self.data_fetcher.fetch_ohlcv_fmp(ticker)
        
        # If FMP fails, try Polygon
        if ohlcv_data is None:
            logger.warning(f"FMP failed for {ticker}, trying Polygon...")
            ohlcv_data = self.data_fetcher.fetch_ohlcv_polygon(ticker)
        
        # If fundamental data missing, try FMP
        if fundamental_data is None:
            logger.warning(f"yfinance fundamental data missing for {ticker}, trying FMP...")
            fundamental_data = self.data_fetcher.fetch_fundamental_fmp(ticker)
        
        # If still no data, generate fallback data
        if ohlcv_data is None:
            logger.warning(f"All APIs failed for {ticker}, generating fallback data...")
            self._generate_fallback_data(ticker)
        else:
            self.ohlcv_data[ticker] = ohlcv_data
        
        if fundamental_data is not None:
            self.fundamental_data[ticker] = fundamental_data
        else:
            self._generate_fallback_fundamental_data(ticker)
        
        if analyst_data is not None:
            self.analyst_estimates[ticker] = analyst_data
        else:
            self._generate_fallback_analyst_data(ticker)
        
        # Generate technical indicators
        self._generate_technical_indicators(ticker)
    
    def _generate_fallback_data(self, ticker: str):
        """Generate fallback OHLCV data when APIs fail"""
        logger.warning(f"Generating fallback OHLCV data for {ticker}")
        
        # Base prices for each stock
        base_prices = {
            'AAPL': 180.0,
            'BABA': 85.0,
            'CAT': 320.0,
            'NVO': 150.0,
            'SIEGY': 80.0
        }
        
        base_price = base_prices.get(ticker, 100.0)
        
        # Generate dates (last 252 trading days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        # Generate realistic price data
        np.random.seed(hash(ticker) % 2**32)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))
        
        # Generate OHLCV data
        ohlcv = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = max(1, int(1000000 * np.random.lognormal(0, 0.3)))
            
            ohlcv.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(price, 2),
                'Volume': volume
            })
        
        self.ohlcv_data[ticker] = pd.DataFrame(ohlcv)
        
        # Store fallback data in cache
        self.data_fetcher.cache.store_ohlcv_data(ticker, self.ohlcv_data[ticker], 'fallback')
    
    def _generate_fallback_fundamental_data(self, ticker: str):
        """Generate fallback fundamental data"""
        logger.warning(f"Generating fallback fundamental data for {ticker}")
        
        current_price = self.ohlcv_data[ticker]['Close'].iloc[-1]
        
        self.fundamental_data[ticker] = {
            'market_cap': np.random.randint(50000000000, 4000000000000),
            'enterprise_value': np.random.randint(60000000000, 4200000000000),
            'pe_ratio': round(np.random.uniform(8.0, 45.0), 2),
            'forward_pe': round(np.random.uniform(7.0, 40.0), 2),
            'peg_ratio': round(np.random.uniform(0.5, 3.5), 2),
            'pb_ratio': round(np.random.uniform(0.8, 8.0), 2),
            'ps_ratio': round(np.random.uniform(1.0, 15.0), 2),
            'debt_to_equity': round(np.random.uniform(0.1, 2.5), 2),
            'current_ratio': round(np.random.uniform(0.8, 3.5), 2),
            'roe': round(np.random.uniform(0.05, 0.35), 3),
            'roa': round(np.random.uniform(0.02, 0.20), 3),
            'revenue_growth': round(np.random.uniform(-0.1, 0.3), 3),
            'earnings_growth': round(np.random.uniform(-0.2, 0.4), 3),
            'dividend_yield': round(np.random.uniform(0.0, 0.05), 3),
            'beta': round(np.random.uniform(0.6, 1.8), 2)
        }
        
        # Store in cache
        self.data_fetcher.cache.store_fundamental_data(ticker, self.fundamental_data[ticker], 'fallback')
    
    def _generate_fallback_analyst_data(self, ticker: str):
        """Generate fallback analyst estimates"""
        logger.warning(f"Generating fallback analyst estimates for {ticker}")
        
        current_price = self.ohlcv_data[ticker]['Close'].iloc[-1]
        
        target_multipliers = {
            'AAPL': {'high': 1.25, 'median': 1.15, 'low': 0.95},
            'BABA': {'high': 1.40, 'median': 1.12, 'low': 0.85},
            'CAT': {'high': 1.20, 'median': 1.08, 'low': 0.90},
            'NVO': {'high': 1.18, 'median': 1.08, 'low': 0.92},
            'SIEGY': {'high': 1.21, 'median': 1.09, 'low': 0.88}
        }
        
        multiplier = target_multipliers.get(ticker, {'high': 1.20, 'median': 1.10, 'low': 0.90})
        
        self.analyst_estimates[ticker] = {
            'current_price': round(current_price, 2),
            'target_high': round(current_price * multiplier['high'], 2),
            'target_median': round(current_price * multiplier['median'], 2),
            'target_low': round(current_price * multiplier['low'], 2),
            'analyst_rating': np.random.uniform(1.5, 2.8),
            'number_of_analysts': np.random.randint(20, 50),
            'current_year_eps_estimate': round(np.random.uniform(2.0, 25.0), 2),
            'next_year_eps_estimate': round(np.random.uniform(2.5, 30.0), 2),
            'current_year_revenue_estimate': np.random.randint(20000000000, 500000000000),
            'next_year_revenue_estimate': np.random.randint(25000000000, 550000000000)
        }
        
        # Store in cache
        self.data_fetcher.cache.store_analyst_estimates(ticker, self.analyst_estimates[ticker], 'fallback')
    
    def _generate_technical_indicators(self, ticker: str):
        """Generate technical analysis indicators"""
        df = self.ohlcv_data[ticker].copy()
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        self.technical_indicators[ticker] = df
    
    def refresh_data(self, ticker: Optional[str] = None, force_refresh: bool = False):
        """Refresh data for specific ticker or all tickers"""
        if ticker:
            tickers = [ticker]
        else:
            tickers = list(self.stocks.keys())
        
        for t in tickers:
            logger.info(f"Refreshing data for {t}...")
            
            if force_refresh:
                # Force fetch fresh data
                self._fetch_fresh_data(t)
            else:
                # Try cache first, then fetch if needed
                self._load_stock_data(t)
            
            time.sleep(2)  # Rate limiting
    
    def get_data_freshness_status(self) -> Dict[str, Dict[str, str]]:
        """Get freshness status of all cached data"""
        status = {}
        
        for ticker in self.stocks.keys():
            status[ticker] = {}
            
            # Check each data type
            for data_type in ['ohlcv', 'fundamental', 'analyst']:
                is_fresh = self.data_fetcher.cache.is_data_fresh(ticker, data_type, self.max_age_hours)
                status[ticker][data_type] = "Fresh" if is_fresh else "Stale"
        
        return status
    
    def test_api_connections(self) -> Dict[str, bool]:
        """Test connections to all configured APIs"""
        results = {}
        
        # Test yfinance
        try:
            stock = yf.Ticker('AAPL')
            info = stock.info
            results['yfinance'] = len(info) > 5
        except:
            results['yfinance'] = False
        
        # Test FMP
        try:
            url = f"{self.data_fetcher.apis['fmp'].base_url}/profile/AAPL"
            params = {'apikey': self.data_fetcher.apis['fmp'].api_key}
            response = requests.get(url, params=params, timeout=5)
            results['fmp'] = response.status_code == 200
        except:
            results['fmp'] = False
        
        # Test Polygon
        try:
            url = f"{self.data_fetcher.apis['polygon'].base_url}/aggs/ticker/AAPL/prev"
            params = {'apikey': self.data_fetcher.apis['polygon'].api_key}
            response = requests.get(url, params=params, timeout=5)
            results['polygon'] = response.status_code == 200
        except:
            results['polygon'] = False
        
        return results
    
    # ==================== EXPORT METHODS (Same as original) ====================
    
    def export_ohlcv_data(self, ticker: str) -> pd.DataFrame:
        """Export OHLCV data for a specific stock."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        return self.ohlcv_data[ticker].copy()
    
    def export_fundamental_data(self, ticker: str) -> Dict:
        """Export fundamental data for a specific stock."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        return self.fundamental_data[ticker].copy()
    
    def export_analyst_estimates(self, ticker: str) -> Dict:
        """Export analyst estimates for a specific stock."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        return self.analyst_estimates[ticker].copy()
    
    def export_technical_indicators(self, ticker: str) -> pd.DataFrame:
        """Export technical indicators DataFrame for a specific stock."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        return self.technical_indicators[ticker].copy()
    
    def export_all_data_for_ticker(self, ticker: str) -> Dict:
        """Export all data for a specific stock."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        
        return {
            'ticker': ticker,
            'company_name': self.stocks[ticker],
            'ohlcv': self.ohlcv_data[ticker].copy(),
            'analyst_estimates': self.analyst_estimates[ticker].copy(),
            'fundamental_data': self.fundamental_data[ticker].copy(),
            'technical_indicators': self.technical_indicators[ticker].copy()
        }
    
    def export_all_stocks_data(self) -> Dict[str, Dict]:
        """Export all data for all stocks."""
        all_data = {}
        for ticker in self.stocks.keys():
            all_data[ticker] = self.export_all_data_for_ticker(ticker)
        return all_data
    
    def get_stock_list(self) -> List[str]:
        """Get list of all available stock tickers."""
        return list(self.stocks.keys())
    
    def get_company_name(self, ticker: str) -> str:
        """Get company name for a ticker."""
        if ticker not in self.stocks:
            raise ValueError(f"Stock {ticker} not found in data")
        return self.stocks[ticker]
    
    def get_basic_summary(self) -> pd.DataFrame:
        """Get basic data summary for all stocks (no analysis calculations)."""
        
        summary_data = []
        for ticker in self.stocks.keys():
            # Get latest price from OHLCV data
            latest_price = self.ohlcv_data[ticker]['Close'].iloc[-1]
            latest_rsi = self.technical_indicators[ticker]['RSI'].iloc[-1]
            
            # Handle None values for market cap and dividend yield
            market_cap = self.fundamental_data[ticker].get('market_cap')
            market_cap_b = round(market_cap / 1e9, 1) if market_cap else None
            
            dividend_yield = self.fundamental_data[ticker].get('dividend_yield')
            div_yield_str = f"{dividend_yield*100:.1f}%" if dividend_yield else "N/A"
            
            summary_data.append({
                'Ticker': ticker,
                'Company': self.stocks[ticker],
                'Current Price': round(latest_price, 2),
                'Target Median': self.analyst_estimates[ticker].get('target_median'),
                'Analyst Rating': round(self.analyst_estimates[ticker].get('analyst_rating', 0), 1) if self.analyst_estimates[ticker].get('analyst_rating') else None,
                'PE Ratio': self.fundamental_data[ticker].get('pe_ratio'),
                'Market Cap (B)': market_cap_b,
                'RSI': round(latest_rsi, 2),
                'Dividend Yield': div_yield_str
            })
        
        return pd.DataFrame(summary_data)
    
    def print_data_overview(self):
        """Print enhanced data overview with freshness status."""
        
        print("="*80)
        print("ENHANCED COMPREHENSIVE STOCK DATA OVERVIEW")
        print("="*80)
        
        print("\n📊 AVAILABLE STOCKS:")
        print("-" * 40)
        for ticker, company in self.stocks.items():
            latest_price = self.ohlcv_data[ticker]['Close'].iloc[-1]
            print(f"{ticker}: {company} - ${latest_price:.2f}")
        
        print(f"\n📈 DATA AVAILABLE:")
        print("-" * 40)
        print("• OHLCV Data: 1 year of daily price/volume data")
        print("• Analyst Estimates: Price targets, EPS estimates, ratings")
        print("• Fundamental Data: PE ratios, growth rates, profitability metrics")
        print("• Technical Indicators: RSI, MACD, Moving Averages, Bollinger Bands")
        
        print(f"\n🔗 API CONNECTION STATUS:")
        print("-" * 40)
        api_status = self.test_api_connections()
        for api, status in api_status.items():
            status_icon = "✓" if status else "✗"
            print(f"{status_icon} {api.upper()}: {'Connected' if status else 'Failed'}")
        
        print(f"\n🔄 DATA FRESHNESS STATUS:")
        print("-" * 40)
        freshness = self.get_data_freshness_status()
        for ticker, status in freshness.items():
            print(f"{ticker}: OHLCV={status['ohlcv']}, Fund={status['fundamental']}, Analyst={status['analyst']}")
        
        print(f"\n🔧 EXPORT METHODS AVAILABLE:")
        print("-" * 40)
        print("• export_ohlcv_data(ticker)")
        print("• export_fundamental_data(ticker)")
        print("• export_analyst_estimates(ticker)")
        print("• export_technical_indicators(ticker)")
        print("• export_all_data_for_ticker(ticker)")
        print("• export_all_stocks_data()")
        
        print(f"\n⚙️ DATA MANAGEMENT:")
        print("-" * 40)
        print("• refresh_data(ticker=None, force_refresh=False)")
        print("• get_data_freshness_status()")
        print("• test_api_connections()")
        print("• Intelligent caching with SQLite backend")
        print("• Multi-source API fallback (yfinance → FMP → Polygon → fallback)")
        print("• Rate limiting and retry mechanisms")
        
        print("\n" + "="*80)

def main():
    """Main function to demonstrate the enhanced data system."""
    
    print("Loading enhanced comprehensive stock data...")
    
    # Initialize with caching enabled
    data = EnhancedComprehensiveStockData(use_cache=True, max_age_hours=24)
    
    # Print data overview
    data.print_data_overview()
    
    # Show basic summary
    print("\n📊 BASIC STOCK SUMMARY:")
    print("-" * 80)
    summary_df = data.get_basic_summary()
    print(summary_df.to_string(index=False))
    
    return data

if __name__ == "__main__":
    comprehensive_data = main()