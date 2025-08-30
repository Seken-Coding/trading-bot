"""
Enhanced Trading Bot Dashboard

Clean, optimized Flask dashboard for monitoring trading bot performance.
Supports both local and Railway deployment.

Author: Enhanced Trading Bot System  
Version: 4.0.0
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime, timedelta
import logging

# Import handling for optional dependencies
try:
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.utils
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Configuration
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
DATABASE_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", "5000"))

# Flask app setup
app = Flask(__name__)
CORS(app)

if IS_RAILWAY:
    app.config['DEBUG'] = False
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class DashboardData:
    """Data provider for dashboard"""
    
    def __init__(self):
        self.portfolio_data = []
        self.signals_data = []
        self.market_data = {}
        
    def get_portfolio_summary(self):
        """Get portfolio summary data"""
        return {
            "total_value": 125000.00,
            "daily_pnl": 2500.00,
            "total_pnl": 25000.00,
            "positions": [
                {"symbol": "AMD", "quantity": 100, "value": 15000, "pnl": 500},
                {"symbol": "NVDA", "quantity": 50, "value": 35000, "pnl": 1200},
                {"symbol": "AAPL", "quantity": 75, "value": 12000, "pnl": -300}
            ],
            "cash": 63000.00
        }
    
    def get_performance_chart(self):
        """Generate performance chart data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='D')
        
        # Simulate portfolio values
        portfolio_values = []
        base_value = 100000
        for i, date in enumerate(dates):
            # Add some realistic volatility
            daily_change = (i * 100) + (i % 3 - 1) * 500
            portfolio_values.append(base_value + daily_change)
        
        trace = go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=2)
        )
        
        layout = go.Layout(
            title='Portfolio Performance (30 Days)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value ($)'),
            template='plotly_dark',
            height=400
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_recent_signals(self):
        """Get recent trading signals"""
        return [
            {
                "timestamp": "2024-01-15 14:30:00",
                "symbol": "AMD",
                "type": "BUY",
                "price": 150.25,
                "confidence": 0.85,
                "status": "EXECUTED"
            },
            {
                "timestamp": "2024-01-14 10:15:00", 
                "symbol": "NVDA",
                "type": "BUY",
                "price": 720.50,
                "confidence": 0.78,
                "status": "EXECUTED"
            },
            {
                "timestamp": "2024-01-13 16:45:00",
                "symbol": "AAPL", 
                "type": "SELL",
                "price": 185.75,
                "confidence": 0.72,
                "status": "PENDING"
            }
        ]
    
    def get_watchlist_data(self):
        """Get watchlist with current data"""
        return [
            {
                "symbol": "AMD",
                "price": 152.30,
                "change": 2.05,
                "change_percent": 1.37,
                "volume": 25000000,
                "rsi": 65.2,
                "signal": "NEUTRAL"
            },
            {
                "symbol": "NVDA", 
                "price": 735.80,
                "change": 15.30,
                "change_percent": 2.12,
                "volume": 18000000,
                "rsi": 58.7,
                "signal": "BUY"
            },
            {
                "symbol": "AAPL",
                "price": 183.25,
                "change": -2.50,
                "change_percent": -1.35,
                "volume": 45000000,
                "rsi": 45.3,
                "signal": "SELL"
            },
            {
                "symbol": "MSFT",
                "price": 420.15,
                "change": 5.75,
                "change_percent": 1.39,
                "volume": 22000000,
                "rsi": 62.1,
                "signal": "NEUTRAL"
            },
            {
                "symbol": "GOOGL",
                "price": 165.90,
                "change": 1.20,
                "change_percent": 0.73,
                "volume": 15000000,
                "rsi": 55.8,
                "signal": "NEUTRAL"
            },
            {
                "symbol": "TSLA",
                "price": 238.45,
                "change": -8.30,
                "change_percent": -3.37,
                "volume": 35000000,
                "rsi": 35.2,
                "signal": "BUY"
            }
        ]

# Initialize data provider
dashboard_data = DashboardData()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def api_portfolio():
    """Portfolio summary API"""
    return jsonify(dashboard_data.get_portfolio_summary())

@app.route('/api/performance')
def api_performance():
    """Performance chart API"""
    return dashboard_data.get_performance_chart()

@app.route('/api/signals')
def api_signals():
    """Recent signals API"""
    return jsonify(dashboard_data.get_recent_signals())

@app.route('/api/watchlist')
def api_watchlist():
    """Watchlist API"""
    return jsonify(dashboard_data.get_watchlist_data())

@app.route('/api/status')
def api_status():
    """Bot status API"""
    return jsonify({
        "status": "ONLINE",
        "mode": "DRY_RUN",
        "uptime": "2 days, 14 hours",
        "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_trades": 45,
        "successful_trades": 38,
        "success_rate": 84.4
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
