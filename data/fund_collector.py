"""
data/fund_collector.py
Collects fund data from Yahoo Finance and calculates technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

class FundDataCollector:
    # Most popular ETFs and mutual funds
    POPULAR_FUNDS = [
        {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'category': 'Large Cap'},
        {'symbol': 'QQQ', 'name': 'Invesco QQQ ETF', 'category': 'Technology'},
        {'symbol': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'category': 'Total Market'},
        {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'category': 'Small Cap'},
        {'symbol': 'EEM', 'name': 'iShares MSCI Emerging Markets ETF', 'category': 'Emerging Markets'},
        {'symbol': 'AGG', 'name': 'iShares Core US Aggregate Bond ETF', 'category': 'Bonds'},
        {'symbol': 'VNQ', 'name': 'Vanguard Real Estate ETF', 'category': 'Real Estate'},
        {'symbol': 'GLD', 'name': 'SPDR Gold Trust', 'category': 'Commodities'},
        {'symbol': 'TLT', 'name': 'iShares 20+ Year Treasury Bond ETF', 'category': 'Bonds'},
        {'symbol': 'XLF', 'name': 'Financial Select Sector SPDR Fund', 'category': 'Financial'},
        {'symbol': 'XLE', 'name': 'Energy Select Sector SPDR Fund', 'category': 'Energy'},
        {'symbol': 'VUG', 'name': 'Vanguard Growth ETF', 'category': 'Growth'},
        {'symbol': 'VTV', 'name': 'Vanguard Value ETF', 'category': 'Value'},
        {'symbol': 'DIA', 'name': 'SPDR Dow Jones Industrial Average ETF', 'category': 'Large Cap'},
        {'symbol': 'IWF', 'name': 'iShares Russell 1000 Growth ETF', 'category': 'Growth'},
    ]
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def initialize_funds(self):
        """Store popular funds in database"""
        self.db.insert_funds(self.POPULAR_FUNDS)
        return self.POPULAR_FUNDS
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the fund"""
        
        # Returns
        df['returns_1w'] = df['Close'].pct_change(5)
        df['returns_1m'] = df['Close'].pct_change(21)
        df['returns_3m'] = df['Close'].pct_change(63)
        df['returns_6m'] = df['Close'].pct_change(126)
        df['returns_1y'] = df['Close'].pct_change(252)
        
        # Volatility (30-day)
        df['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
        
        # Sharpe Ratio (simplified, assuming risk-free rate = 0)
        returns = df['Close'].pct_change()
        df['sharpe_ratio'] = (returns.rolling(window=252).mean() * 252) / \
                             (returns.rolling(window=252).std() * np.sqrt(252))
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        return df
    
    def fetch_fund_data(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """Fetch historical data for a fund"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"  ⚠ No data for {symbol}")
                return None
            
            df = self.calculate_technical_indicators(df)
            df['symbol'] = symbol
            
            return df
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
            return None
    
    def prepare_metrics_for_db(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Convert DataFrame to database-ready format"""
        metrics = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['returns_1w']):  # Skip incomplete data
                continue
            
            metric = {
                'fund_symbol': symbol,
                'date': idx.strftime('%Y-%m-%d'),
                'close_price': float(row['Close']) if not pd.isna(row['Close']) else None,
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else None,
                'returns_1w': float(row['returns_1w']) if not pd.isna(row['returns_1w']) else None,
                'returns_1m': float(row['returns_1m']) if not pd.isna(row['returns_1m']) else None,
                'returns_3m': float(row['returns_3m']) if not pd.isna(row['returns_3m']) else None,
                'returns_6m': float(row['returns_6m']) if not pd.isna(row['returns_6m']) else None,
                'returns_1y': float(row['returns_1y']) if not pd.isna(row['returns_1y']) else None,
                'volatility_30d': float(row['volatility_30d']) if not pd.isna(row['volatility_30d']) else None,
                'sharpe_ratio': float(row['sharpe_ratio']) if not pd.isna(row['sharpe_ratio']) else None,
                'rsi_14': float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None,
                'sma_50': float(row['sma_50']) if not pd.isna(row['sma_50']) else None,
                'sma_200': float(row['sma_200']) if not pd.isna(row['sma_200']) else None,
            }
            metrics.append(metric)
        
        return metrics
    
    def collect_all_funds(self):
        """Main collection workflow"""
        # Initialize funds in database
        print("Initializing fund list...")
        self.initialize_funds()
        
        all_metrics = []
        
        for fund in self.POPULAR_FUNDS:
            symbol = fund['symbol']
            print(f"\nFetching {symbol} - {fund['name']}...")
            
            df = self.fetch_fund_data(symbol)
            
            if df is not None:
                metrics = self.prepare_metrics_for_db(df, symbol)
                all_metrics.extend(metrics)
                print(f"  ✓ Collected {len(metrics)} records")
        
        # Store all metrics
        if all_metrics:
            print(f"\nStoring {len(all_metrics)} total metric records...")
            self.db.insert_fund_metrics(all_metrics)
        
        return all_metrics