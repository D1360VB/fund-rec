# send_telegram.py
import os
import requests
from database.db_manager import DatabaseManager
from dotenv import load_dotenv

load_dotenv()


def format_recommendations(recs):
    """Format recommendations as a Telegram message in English."""
    message = "📈 *Weekly Recommendations*\n\n"
    for rec in recs:
        emoji = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '⚪',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }.get(rec['recommendation'], '⚪')
        
        label = {
            'STRONG_BUY': 'Strong Buy',
            'BUY': 'Buy',
            'HOLD': 'Hold',
            'SELL': 'Sell',
            'STRONG_SELL': 'Strong Sell'
        }.get(rec['recommendation'], 'Hold')
        
        message += f"{emoji} *{rec['fund_symbol']}*: {label} "
        message += f"(Confidence: {rec['confidence']*100:.1f}%)\n"
    
    message += f"\n_Updated: {recs[0]['week_start']}_"
    return message


def main():
    db = DatabaseManager()
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("✗ Missing Telegram credentials")
        return
    
    # Obtener las recomendaciones más recientes
    recs = db.get_latest_recommendations()
    
    if not recs:
        print("✗ No recommendations to send")
        return
    
    # Filtrar solo las de la semana más reciente
    latest_week = recs[0]['week_start']
    week_recs = [r for r in recs if r['week_start'] == latest_week]
    
    # Deduplicar: quedarnos con la más reciente por fondo
    seen = set()
    latest_recs = []
    for r in week_recs:
        if r['fund_symbol'] not in seen:
            seen.add(r['fund_symbol'])
            latest_recs.append(r)
    
    # Ordenar alfabéticamente
    latest_recs.sort(key=lambda r: r['fund_symbol'])
    
    print(f"📊 Sending {len(latest_recs)} recommendations for week {latest_week}")
    
    message = format_recommendations(latest_recs)
    
    # Enviar a Telegram
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("✓ Message sent to Telegram")
    else:
        print(f"✗ Error: {response.text}")


if __name__ == '__main__':
    main()
