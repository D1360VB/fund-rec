"""
models/recommender.py
Machine Learning model for fund recommendations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
from typing import List, Dict
import os

class FundRecommender:
    RECOMMENDATIONS = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'returns_1w', 'returns_1m', 'returns_3m', 'returns_6m', 'returns_1y',
            'volatility_30d', 'sharpe_ratio', 'rsi_14', 
            'price_to_sma50', 'price_to_sma200', 'sma50_to_sma200'
        ]
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels based on future returns
        Look ahead 1 week to determine if the fund went up/down
        """
        # Calculate future 1-week return
        df = df.sort_values('date')
        df['future_return'] = df.groupby('fund_symbol')['close_price'].shift(-5).pct_change(5)
        
        # Create recommendation labels based on future returns
        conditions = [
            df['future_return'] <= -0.03,  # Strong Sell: < -3%
            (df['future_return'] > -0.03) & (df['future_return'] <= -0.01),  # Sell: -3% to -1%
            (df['future_return'] > -0.01) & (df['future_return'] < 0.01),   # Hold: -1% to 1%
            (df['future_return'] >= 0.01) & (df['future_return'] < 0.03),   # Buy: 1% to 3%
            df['future_return'] >= 0.03,   # Strong Buy: > 3%
        ]
        
        labels = np.select(conditions, [0, 1, 2, 3, 4], default=2)
        return labels
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        df = df.copy()
        
        # Price relative to moving averages
        df['price_to_sma50'] = df['close_price'] / df['sma_50']
        df['price_to_sma200'] = df['close_price'] / df['sma_200']
        df['sma50_to_sma200'] = df['sma_50'] / df['sma_200']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def prepare_training_data(self, metrics: List[Dict]) -> tuple:
        """Prepare data for model training"""
        df = pd.DataFrame(metrics)
        
        if df.empty:
            raise ValueError("No metrics data available for training")
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create labels
        df['label'] = self.create_labels(df)
        
        # Remove rows without labels (last few rows of each fund)
        df = df.dropna(subset=['label'])
        
        # Select features
        X = df[self.feature_columns]
        y = df['label']
        
        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        return X, y, df
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Fetching training data from database...")
        metrics = self.db.get_all_metrics_for_training()
        
        if not metrics:
            print("✗ No data available for training")
            return
        
        print(f"Preparing {len(metrics)} records...")
        X, y, _ = self.prepare_training_data(metrics)
        
        print(f"Training on {len(X)} samples...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Save model
        os.makedirs('models/saved', exist_ok=True)
        joblib.dump(self.model, 'models/saved/rf_model.pkl')
        joblib.dump(self.scaler, 'models/saved/scaler.pkl')
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n✓ Model trained successfully")
        print("\nTop 5 Important Features:")
        print(feature_importance.head().to_string(index=False))
        
        return self.model
    
    def load_model(self):
        """Load saved model"""
        try:
            self.model = joblib.load('models/saved/rf_model.pkl')
            self.scaler = joblib.load('models/saved/scaler.pkl')
            return True
        except:
            return False
    
    def predict_single_fund(self, symbol: str) -> Dict:
        """Generate recommendation for a single fund"""
        # Get latest metrics
        metrics = self.db.get_fund_metrics(symbol, days=300)
        
        if not metrics:
            return None
        
        df = pd.DataFrame(metrics)
        df = self.engineer_features(df)
        
        # Get most recent complete record
        latest = df[df[self.feature_columns].notna().all(axis=1)].iloc[0]
        
        # Prepare features
        X = latest[self.feature_columns].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        recommendation = self.RECOMMENDATIONS[int(prediction)]
        confidence = probabilities[int(prediction)]
        
        return {
            'fund_symbol': symbol,
            'recommendation': recommendation,
            'confidence': float(confidence),
            'current_price': float(latest['close_price']),
            'date': latest['date']
        }
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate recommendations for all funds"""
        # Load or train model
        if not self.load_model():
            print("No saved model found, training new model...")
            self.train_model()
        
        funds = self.db.get_all_funds()
        recommendations = []
        week_start = datetime.now().date()
        
        for fund in funds:
            symbol = fund['symbol']
            print(f"Analyzing {symbol}...", end=' ')
            
            rec = self.predict_single_fund(symbol)
            
            if rec:
                rec['week_start'] = week_start.strftime('%Y-%m-%d')
                rec['fund_name'] = fund['name']
                recommendations.append(rec)
                print(f"✓ {rec['recommendation']}")
            else:
                print("✗ Insufficient data")
        
        # Save recommendations to database
        if recommendations:
            db_recs = [{k: v for k, v in rec.items() 
                       if k in ['fund_symbol', 'recommendation', 'confidence', 
                               'current_price', 'week_start']} 
                      for rec in recommendations]
            self.db.insert_recommendations(db_recs)
        
        return recommendations