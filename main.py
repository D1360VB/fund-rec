"""
Fund Recommendation System MVP
Main orchestration script
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from database.db_manager import DatabaseManager
from data.fund_collector import FundDataCollector
from models.recommender import FundRecommender

# Load environment variables from .env file
load_dotenv()

def main():
    """Main execution flow"""
    
    # Initialize components
    db = DatabaseManager()
    collector = FundDataCollector(db)
    recommender = FundRecommender(db)
    
    print("=" * 60)
    print("FUND RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Step 1: Collect and store fund data
    print("\n[1/3] Collecting fund data from Yahoo Finance...")
    collector.collect_all_funds()
    
    # Step 2: Train/update ML model
    print("\n[2/3] Training ML model...")
    recommender.train_model()
    
    # Step 3: Generate recommendations
    print("\n[3/3] Generating recommendations...")
    recommendations = recommender.generate_recommendations()
    
    # Display results
    print("\n" + "=" * 60)
    print("WEEKLY RECOMMENDATIONS")
    print("=" * 60)
    for rec in recommendations:
        print(f"\n{rec['fund_symbol']:8} - {rec['fund_name']}")
        print(f"  Recommendation: {rec['recommendation']:12} (Confidence: {rec['confidence']:.1%})")
        print(f"  Current Price: ${rec['current_price']:.2f}")
    
    print("\n✓ Recommendations saved to database")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()