# 📈 Weekly Fund Recommendation System

An automated machine learning system that generates weekly investment recommendations (Strong Buy, Buy, Hold, Sell, Strong Sell) for popular ETFs and mutual funds.

## 🎯 Features

- **Automated Data Collection**: Fetches historical data for 15 popular ETFs from Yahoo Finance
- **Technical Analysis**: Calculates 11 technical indicators (RSI, moving averages, volatility, Sharpe ratio, etc.)
- **ML Predictions**: Random Forest classifier trained on historical patterns
- **Weekly Automation**: Runs automatically via cron job
- **Database Storage**: All data and recommendations stored in Supabase
- **Confidence Scores**: Each recommendation includes probability score

## 📊 Covered Funds

- **Large Cap**: SPY, DIA
- **Technology**: QQQ
- **Total Market**: VTI
- **Small Cap**: IWM
- **Growth/Value**: VUG, VTV, IWF
- **Sectors**: XLF (Financial), XLE (Energy)
- **International**: EEM (Emerging Markets)
- **Bonds**: AGG, TLT
- **Alternatives**: VNQ (Real Estate), GLD (Gold)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account (free tier works)
- Debian/Ubuntu Linux (or similar)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weekly-fund-rec.git
cd fund-rec
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Supabase:
   - Create a project at [supabase.com](https://supabase.com)
   - Run the SQL schema from `database/db_manager.py` in SQL Editor
   - Get your Project URL and API key

5. Configure environment variables:
```bash
cp .env.example .env
nano .env
```
Add your Supabase credentials: