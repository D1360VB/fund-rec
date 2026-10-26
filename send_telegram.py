# send_telegram.py
import os
import requests
from database.db_manager import DatabaseManager
from dotenv import load_dotenv

load_dotenv()

def format_recommendations(recs):
    """Formatea las recomendaciones como un mensaje de Telegram."""
    message = "📈 *Recomendaciones de la semana*\n\n"
    for rec in recs:
        # Emoji según la recomendación
        emoji = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '⚪',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }.get(rec['recommendation'], '⚪')
        
        # Traducción
        label = {
            'STRONG_BUY': 'Fuerte Compra',
            'BUY': 'Compra',
            'HOLD': 'Mantener',
            'SELL': 'Venta',
            'STRONG_SELL': 'Fuerte Venta'
        }.get(rec['recommendation'], 'Mantener')
        
        message += f"{emoji} *{rec['fund_symbol']}*: {label} "
        message += f"(Confianza: {rec['confidence']*100:.1f}%)\n"
    
    message += f"\n_Actualizado: {recs[0]['week_start']}_"
    return message

def main():
    db = DatabaseManager()
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("✗ Faltan credenciales de Telegram")
        return
    
    # Obtener las recomendaciones más recientes (las de esta semana)
    recs = db.get_latest_recommendations()
    
    if not recs:
        print("✗ No hay recomendaciones para enviar")
        return
    
    # Filtrar solo las de la semana más reciente
    latest_week = recs[0]['week_start']
    latest_recs = [r for r in recs if r['week_start'] == latest_week]
    
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
        print("✓ Mensaje enviado a Telegram")
    else:
        print(f"✗ Error: {response.text}")

if __name__ == '__main__':
    main()
