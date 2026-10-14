"""
evaluate_recommendations.py
Evalúa las recomendaciones pasadas midiendo el retorno real a 1 semana.
Se debe ejecutar después de que las recomendaciones tengan ≥5 días.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from database.db_manager import DatabaseManager
import pandas as pd  

load_dotenv()


# Umbrales para decidir si una recomendación fue "correcta"
# Ajusta según tu criterio
HOLD_THRESHOLD = 0.01  # ±1% se considera HOLD acertado


def is_recommendation_correct(recommendation: str, actual_return: float) -> bool:
    """
    Determina si la recomendación fue correcta según el retorno real.
    
    Lógica:
      - STRONG_BUY / BUY  → correcto si el retorno fue positivo
      - HOLD              → correcto si el retorno estuvo dentro de ±1%
      - SELL / STRONG_SELL → correcto si el retorno fue negativo
    """
    rec = recommendation.upper().replace(' ', '_')
    
    if rec in ('STRONG_BUY', 'BUY'):
        return actual_return > 0
    elif rec == 'HOLD':
        return abs(actual_return) < HOLD_THRESHOLD
    elif rec in ('SELL', 'STRONG_SELL'):
        return actual_return < 0
    else:
        return False


def evaluate_all_pending(db: DatabaseManager):
    """Evalúa todas las recomendaciones pendientes."""
    print("=" * 60)
    print("EVALUACIÓN DE RECOMENDACIONES PASADAS")
    print("=" * 60)
    
    pending = db.get_unevaluated_recommendations(min_days_old=5)
    
    if not pending:
        print("✓ No hay recomendaciones pendientes de evaluar.")
        return
    
    print(f"Encontradas {len(pending)} recomendaciones para evaluar.\n")
    
    outcomes = []
    
    for rec in pending:
        symbol = rec['fund_symbol']
        week_start = rec['week_start']
        recommendation = rec['recommendation']
        confidence = rec.get('confidence')
        price_at_rec = rec.get('current_price')
        
        print(f"Evaluando {symbol} ({week_start}) → {recommendation}...", end=' ')
        
        target_date = pd.bdate_range(
            start=week_start, periods=6
        )[-1].date().isoformat()
        
        price_after = db.get_price_on_date(symbol, target_date)
        
        if price_after is None or price_at_rec is None:
            print("✗ Sin datos de precio")
            continue
        
        actual_return = (price_after - price_at_rec) / price_at_rec
        correct = is_recommendation_correct(recommendation, actual_return)
        
        outcomes.append({
            'recommendation_id': rec['id'],
            'fund_symbol': symbol,
            'week_start': week_start,
            'evaluation_date': target_date,
            'recommendation': recommendation,
            'confidence': confidence,
            'price_at_recommendation': price_at_rec,
            'price_after_1w': price_after,
            'actual_return_1w': actual_return,
            'is_correct': correct,
        })
        
        icon = "✓" if correct else "✗"
        print(f"{icon} retorno={actual_return*100:+.2f}%")
    
    if outcomes:
        if db.insert_recommendation_outcomes(outcomes):
            correct_count = sum(1 for o in outcomes if o['is_correct'])
            accuracy = correct_count / len(outcomes) * 100
            print(f"\n✓ {len(outcomes)} outcomes guardados.")
            print(f"  Precisión de esta tanda: {accuracy:.1f}%")
        else:
            print("\n✗ Error al guardar outcomes.")
    else:
        print("\n✗ No se pudo evaluar ninguna recomendación.")


if __name__ == '__main__':
    db = DatabaseManager()
    evaluate_all_pending(db)
