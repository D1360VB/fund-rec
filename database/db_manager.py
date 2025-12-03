"""
database/db_manager.py
Handles all Supabase database operations
"""

import os
from supabase import create_client, Client
from datetime import datetime
from typing import List, Dict

class DatabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
    
    def setup_tables(self):
        """
        Create necessary tables in Supabase
        Run this SQL in Supabase SQL Editor:
        
        -- Popular funds master list
        CREATE TABLE IF NOT EXISTS funds (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE NOT NULL,
            name VARCHAR(200) NOT NULL,
            category VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Historical fund metrics
        CREATE TABLE IF NOT EXISTS fund_metrics (
            id SERIAL PRIMARY KEY,
            fund_symbol VARCHAR(10) REFERENCES funds(symbol),
            date DATE NOT NULL,
            close_price DECIMAL(10,2),
            volume BIGINT,
            returns_1w DECIMAL(8,4),
            returns_1m DECIMAL(8,4),
            returns_3m DECIMAL(8,4),
            returns_6m DECIMAL(8,4),
            returns_1y DECIMAL(8,4),
            volatility_30d DECIMAL(8,4),
            sharpe_ratio DECIMAL(8,4),
            rsi_14 DECIMAL(8,4),
            sma_50 DECIMAL(10,2),
            sma_200 DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(fund_symbol, date)
        );
        
        -- Weekly recommendations
        CREATE TABLE IF NOT EXISTS recommendations (
            id SERIAL PRIMARY KEY,
            fund_symbol VARCHAR(10) REFERENCES funds(symbol),
            recommendation VARCHAR(20) NOT NULL,
            confidence DECIMAL(5,4),
            current_price DECIMAL(10,2),
            week_start DATE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Create indexes
        CREATE INDEX idx_fund_metrics_symbol ON fund_metrics(fund_symbol);
        CREATE INDEX idx_fund_metrics_date ON fund_metrics(date);
        CREATE INDEX idx_recommendations_week ON recommendations(week_start);
        """
        print("Please run the SQL schema in Supabase SQL Editor (see docstring)")
    
    def insert_funds(self, funds: List[Dict]):
        """Insert popular funds into database"""
        try:
            response = self.supabase.table('funds').upsert(
                funds, 
                on_conflict='symbol'
            ).execute()
            print(f"✓ Inserted {len(funds)} funds")
            return response
        except Exception as e:
            print(f"✗ Error inserting funds: {e}")
            raise
    
    def insert_fund_metrics(self, metrics: List[Dict]):
        """Insert fund metrics data"""
        try:
            response = self.supabase.table('fund_metrics').upsert(
                metrics,
                on_conflict='fund_symbol,date'
            ).execute()
            print(f"✓ Inserted {len(metrics)} metric records")
            return response
        except Exception as e:
            print(f"✗ Error inserting metrics: {e}")
            raise
    
    def get_all_funds(self) -> List[Dict]:
        """Retrieve all funds"""
        response = self.supabase.table('funds').select('*').execute()
        return response.data
    
    def get_fund_metrics(self, symbol: str = None, days: int = 365) -> List[Dict]:
        """Retrieve fund metrics for training"""
        query = self.supabase.table('fund_metrics').select('*')
        
        if symbol:
            query = query.eq('fund_symbol', symbol)
        
        query = query.order('date', desc=True).limit(days)
        response = query.execute()
        return response.data
    
    def get_all_metrics_for_training(self) -> List[Dict]:
        """Get all recent metrics for model training"""
        # Get last 100 days of data for all funds
        response = self.supabase.table('fund_metrics')\
            .select('*')\
            .order('date', desc=True)\
            .limit(10000)\
            .execute()
        return response.data
    
    def insert_recommendations(self, recommendations: List[Dict]):
        """Insert weekly recommendations"""
        try:
            response = self.supabase.table('recommendations').insert(
                recommendations
            ).execute()
            print(f"✓ Inserted {len(recommendations)} recommendations")
            return response
        except Exception as e:
            print(f"✗ Error inserting recommendations: {e}")
            raise
    
    def get_latest_recommendations(self) -> List[Dict]:
        """Get most recent recommendations"""
        response = self.supabase.table('recommendations')\
            .select('*, funds(name)')\
            .order('created_at', desc=True)\
            .limit(50)\
            .execute()
        return response.data