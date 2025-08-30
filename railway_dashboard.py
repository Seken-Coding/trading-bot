#!/usr/bin/env python3
"""
Railway Dashboard - Lightweight Web Interface

Memory-optimized dashboard for Railway deployment.
"""

from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

# Railway configuration
DATABASE_URL = os.getenv("DATABASE_URL")
POSTGRES_HOST = os.getenv("PGHOST", "localhost")
POSTGRES_PORT = int(os.getenv("PGPORT", "5432"))
POSTGRES_DB = os.getenv("PGDATABASE", "railway")
POSTGRES_USER = os.getenv("PGUSER", "postgres")
POSTGRES_PASSWORD = os.getenv("PGPASSWORD", "")

class DataProvider:
    """Lightweight data provider for dashboard"""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to Railway PostgreSQL"""
        try:
            if DATABASE_URL:
                self.connection = psycopg2.connect(DATABASE_URL)
            else:
                self.connection = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD
                )
        except Exception as e:
            print(f"Database connection failed: {e}")
    
    def get_recent_activities(self, limit: int = 10):
        """Get recent trading activities"""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT symbol, action, data, timestamp 
                    FROM trading_activities 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Query failed: {e}")
            return []
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        return {
            "total_value": 125000.00,
            "daily_pnl": 2500.00,
            "total_pnl": 25000.00,
            "cash": 63000.00,
            "positions": [
                {"symbol": "AMD", "quantity": 100, "value": 15000, "pnl": 500},
                {"symbol": "NVDA", "quantity": 50, "value": 35000, "pnl": 1200}
            ]
        }
    
    def get_bot_status(self):
        """Get bot status"""
        return {
            "status": "ONLINE",
            "mode": os.getenv("DRY_RUN", "true").upper(),
            "uptime": "Running on Railway",
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "environment": "Railway Cloud",
            "memory_limit": f"{os.getenv('RAILWAY_MEMORY_LIMIT', '512')}MB"
        }

data_provider = DataProvider()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('railway_dashboard.html')

@app.route('/health')
def health():
    """Health check for Railway"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.route('/api/portfolio')
def api_portfolio():
    """Portfolio API"""
    return jsonify(data_provider.get_portfolio_summary())

@app.route('/api/activities')
def api_activities():
    """Recent activities API"""
    activities = data_provider.get_recent_activities()
    return jsonify([dict(activity) for activity in activities])

@app.route('/api/status')
def api_status():
    """Bot status API"""
    return jsonify(data_provider.get_bot_status())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
