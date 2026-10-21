"""
backfill_recommendations.py
Genera recomendaciones retroactivas para las últimas N semanas
y las evalúa, para poblar el histórico de rendimiento.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from database.db_manager import DatabaseManager
from models.recommender import FundRecommender

load_dotenv()


def get_business_days_back(n_weeks: int) -> list:
    """
    Devuelve una lista de fechas (lunes) para las últimas n_weeks semanas,
    excluyendo la semana actual.
    """
    today = datetime.now().date()
    # Encontrar el lunes de esta semana
    monday_this_week = today - timedelta(days=today.weekday())
    
    dates = []
    for i in range(n_weeks, 0, -1):
        week_start = monday_this_week - timedelta(weeks=i)
        dates.append(week_start.isoformat())
    return dates


def backfill(db: DatabaseManager, recommender: FundRecommender, n_weeks: int = 6):
    """
    Para cada una de las últimas n_weeks semanas:
    1. Obtiene los datos de fund_metrics hasta esa fecha
    2. Entrena un modelo con esos datos (o usa el actual)
    3. Genera recomendaciones como si fuera esa fecha
    4. Las inserta en recommendations con el week_start correcto
    """
    print("=" * 60)
    print(f"BACKFILL: Generando recomendaciones para {n_weeks} semanas")
    print("=" * 60)
    
    week_dates = get_business_days_back(n_weeks)
    funds = db.get_all_funds()
    
    total_inserted = 0
    
    for week_start in week_dates:
        print(f"\n📅 Semana del {week_start}")
        
        # Verificar si ya hay recomendaciones para esa semana
        existing = db.supabase.table('recommendations') \
            .select('id') \
            .eq('week_start', week_start) \
            .execute()
        
        if existing.data:
            print(f"  ⏭ Ya existen {len(existing.data)} recomendaciones, se omite")
            continue
        
        # Para cada fondo, generar recomendación usando datos hasta esa fecha
        recommendations = []
        for fund in funds:
            symbol = fund['symbol']
            
            # Obtener métricas hasta esa fecha
            metrics = db.supabase.table('fund_metrics') \
                .select('*') \
                .eq('fund_symbol', symbol) \
                .lte('date', week_start) \
                .order('date', desc=True) \
                .limit(300) \
                .execute()
            
            if not metrics.data or len(metrics.data) < 200:
                print(f"  ⚠ {symbol}: datos insuficientes")
                continue
            
            df = pd.DataFrame(metrics.data)
            df = df.sort_values('date').reset_index(drop=True)
            df = recommender.engineer_features(df)
            
            # Última fila completa
            valid = df[recommender.feature_columns].notna().all(axis=1)
            if not valid.any():
                print(f"  ⚠ {symbol}: sin features válidos")
                continue
            
            latest = df[valid].iloc[-1]
            X = latest[recommender.feature_columns].values.reshape(1, -1)
            X_scaled = recommender.scaler.transform(X)
            
            prediction = recommender.model.predict(X_scaled)[0]
            probabilities = recommender.model.predict_proba(X_scaled)[0]
            
            recommendations.append({
                'fund_symbol': symbol,
                'recommendation': recommender.RECOMMENDATIONS[int(prediction)],
                'confidence': float(probabilities[int(prediction)]),
                'current_price': float(latest['close_price']),
                'week_start': week_start,
            })
        
        if recommendations:
            db.supabase.table('recommendations').insert(recommendations).execute()
            print(f"  ✓ Insertadas {len(recommendations)} recomendaciones")
            total_inserted += len(recommendations)
    
    print(f"\n✅ Total: {total_inserted} recomendaciones retroactivas insertadas")


def evaluate_all(db: DatabaseManager):
    """Ejecuta la evaluación de todas las recomendaciones pendientes."""
    from evaluate_recommendations import evaluate_all_pending
    evaluate_all_pending(db)


if __name__ == '__main__':
    db = DatabaseManager()
    recommender = FundRecommender(db)
    
    if not recommender.load_model():
        print("✗ No hay modelo entrenado. Ejecuta main.py primero.")
        exit(1)
    
    backfill(db, recommender, n_weeks=6)
    
    print("\n" + "=" * 60)
    print("EVALUANDO RECOMENDACIONES RETROACTIVAS")
    print("=" * 60)
    evaluate_all(db)
