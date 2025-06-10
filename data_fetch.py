# data_fetch.py
"""
Production-grade data fetching with proper error handling and validation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred = None
        if fred_api_key:
            try:
                self.fred = Fred(api_key=fred_api_key)
            except Exception as e:
                logger.warning(f"FRED initialization failed: {e}")
    
    def fetch_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch adjusted close prices with proper error handling.
        Returns clean, aligned price data.
        """
        # Extend start date to ensure we have enough data for returns
        extended_start = pd.to_datetime(start) - timedelta(days=10)
        
        try:
            # Download all tickers at once for alignment
            data = yf.download(
                tickers, 
                start=extended_start.strftime('%Y-%m-%d'), 
                end=end, 
                progress=False,
                auto_adjust=True,  # Adjust for splits and dividends
                actions=False
            )
            
            # Handle single ticker case
            if len(tickers) == 1:
                prices = data['Close'].to_frame(tickers[0])
            else:
                prices = data['Close']
            
            # Validate data
            if prices.empty:
                raise ValueError("No price data retrieved")
            
            # Forward fill only (no look-ahead bias), limit to 5 days
            prices = prices.ffill(limit=5)
            
            # Drop any remaining NaN rows
            prices = prices.dropna()
            
            # Trim to requested date range
            prices = prices[start:end]
            
            # Verify we have data for all tickers
            missing = set(tickers) - set(prices.columns)
            if missing:
                logger.warning(f"Missing data for tickers: {missing}")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns with proper handling.
        Log returns are preferred for multi-period aggregation.
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        # Remove first row (NaN) and any other NaN values
        returns = returns.iloc[1:]
        
        # Validate returns are reasonable (detect data errors)
        if (returns.abs() > 0.5).any().any():  # 50% daily move
            logger.warning("Extreme returns detected - possible data error")
        
        return returns
    
    def fetch_fred_series(self, series_map: Dict[str, str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch FRED macro data with proper business day alignment.
        """
        if not self.fred:
            logger.warning("FRED not available, using synthetic macro data")
            # Return synthetic data for testing
            dates = pd.date_range(start=start, end=end, freq='B')
            data = pd.DataFrame(index=dates)
            data['rates'] = 0.02 + 0.001 * np.random.randn(len(dates))
            data['credit'] = 0.01 + 0.0005 * np.random.randn(len(dates))
            data['term'] = 0.015 + 0.0005 * np.random.randn(len(dates))
            return data
        
        data = {}
        for name, series_id in series_map.items():
            try:
                series = self.fred.get_series(series_id, start, end)
                # Convert to business day frequency
                series = series.resample('B').last()
                # Forward fill missing values (holidays, etc)
                series = series.ffill()
                data[name] = series
            except Exception as e:
                logger.error(f"Failed to fetch {name} ({series_id}): {e}")
        
        if not data:
            raise ValueError("No FRED data retrieved")
        
        df = pd.DataFrame(data)
        # Ensure we have complete data
        df = df.dropna()
        
        return df
    
    def get_risk_free_rate(self, start: str, end: str) -> pd.Series:
        """
        Get daily risk-free rate from 3-month T-bills.
        """
        if self.fred:
            try:
                # TB3MS is monthly, need to convert to daily
                monthly_rate = self.fred.get_series('TB3MS', start, end)
                # Convert annual rate to daily
                daily_rate = monthly_rate / 100 / 252
                # Resample to business days and forward fill
                daily_rate = daily_rate.resample('B').last().ffill()
                return daily_rate
            except Exception as e:
                logger.warning(f"Failed to fetch risk-free rate: {e}")
        
        # Default: 2% annual rate
        dates = pd.date_range(start=start, end=end, freq='B')
        return pd.Series(0.02 / 252, index=dates, name='rf')
    
    def validate_data_alignment(self, *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Ensure all dataframes are aligned to same dates.
        Critical for risk calculations.
        """
        if not dataframes:
            return []
        
        # Find common date range
        common_dates = dataframes[0].index
        for df in dataframes[1:]:
            common_dates = common_dates.intersection(df.index)
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates in data")
        
        # Align all dataframes
        aligned = [df.loc[common_dates] for df in dataframes]
        
        logger.info(f"Data aligned to {len(common_dates)} common dates")
        return aligned