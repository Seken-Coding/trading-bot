"""
Enhanced Multi-Stock Swing Trading Bot with Discord Integration

Features:
- Multi-stock support with configurable watchlist
- Discord webhook notifications for all trading activities
- PostgreSQL integration for activity logging
- Modern web dashboard for data visualization
- Risk management and portfolio tracking
- Real-time monitoring and alerts

Author: Enhanced Trading Bot System
Version: 3.0.0
"""

import os
import time
import math
import logging
import signal
import sys
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

import requests
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Union

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

# Configuration with environment variables
WATCHLIST = os.getenv("WATCHLIST", "AMD,NVDA,AAPL,MSFT,GOOGL,TSLA").split(",")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Discord Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_NOTIFICATIONS = {
    "purchases": os.getenv("DISCORD_NOTIFY_PURCHASES", "true").lower() == "true",
    "changes": os.getenv("DISCORD_NOTIFY_CHANGES", "true").lower() == "true",
    "daily_updates": os.getenv("DISCORD_NOTIFY_DAILY", "true").lower() == "true",
    "service_status": os.getenv("DISCORD_NOTIFY_STATUS", "true").lower() == "true",
    "recommendations": os.getenv("DISCORD_NOTIFY_RECOMMENDATIONS", "true").lower() == "true"
}

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_bot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "trading_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Trading Configuration
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITIONS_PER_STOCK = int(os.getenv("MAX_POSITIONS_PER_STOCK", "1"))
MAX_TOTAL_POSITIONS = int(os.getenv("MAX_TOTAL_POSITIONS", "5"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("enhanced_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_trading_bot")

# Global shutdown flag
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class NotificationType(Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    DAILY_UPDATE = "daily_update"
    SERVICE_STATUS = "service_status"
    RECOMMENDATION = "recommendation"
    ERROR = "error"
    WARNING = "warning"

@dataclass
class TradeSignal:
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    confidence: float
    fib_level: str
    rsi: float
    sma50: float
    timestamp: datetime

@dataclass
class PortfolioPosition:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

class DiscordNotifier:
    """Discord webhook notification system"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session = None
    
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def send_notification(self, notification_type: NotificationType, 
                              title: str, description: str, 
                              fields: Optional[List[Dict]] = None, color: Optional[int] = None):
        """Send Discord notification"""
        if not self.webhook_url or not DISCORD_NOTIFICATIONS.get(notification_type.value, True):
            return
        
        await self.init_session()
        
        # Color scheme
        colors = {
            NotificationType.PURCHASE: 0x00ff00,  # Green
            NotificationType.SALE: 0xff9900,     # Orange
            NotificationType.DAILY_UPDATE: 0x0099ff,  # Blue
            NotificationType.SERVICE_STATUS: 0x9900ff,  # Purple
            NotificationType.RECOMMENDATION: 0xffff00,  # Yellow
            NotificationType.ERROR: 0xff0000,    # Red
            NotificationType.WARNING: 0xff6600   # Dark Orange
        }
        
        embed = {
            "title": title,
            "description": description,
            "color": color or colors.get(notification_type, 0x808080),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Enhanced Trading Bot v3.0"}
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {
            "embeds": [embed],
            "username": "Trading Bot"
        }
        
        try:
            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:
                    logger.debug(f"Discord notification sent: {title}")
                else:
                    logger.warning(f"Discord notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
    
    def send_sync(self, notification_type: NotificationType, 
                  title: str, description: str, 
                  fields: Optional[List[Dict]] = None, color: Optional[int] = None):
        """Synchronous wrapper for send_notification"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                asyncio.create_task(self.send_notification(
                    notification_type, title, description, fields, color
                ))
            else:
                # If no loop is running, run until complete
                loop.run_until_complete(self.send_notification(
                    notification_type, title, description, fields, color
                ))
        except Exception as e:
            logger.error(f"Discord sync notification error: {e}")

class DatabaseManager:
    """PostgreSQL database manager for activity logging"""
    
    def __init__(self):
        self.connection = None
        self.setup_tables()
    
    def connect(self):
        """Establish database connection"""
        if not psycopg2:
            logger.warning("psycopg2 not available, database features disabled")
            return False
        
        try:
            self.connection = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def setup_tables(self):
        """Create database tables if they don't exist"""
        if not psycopg2 or not self.connect():
            logger.warning("Database setup skipped - PostgreSQL not available")
            return
        
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                # Trading activities table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_activities (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        symbol VARCHAR(10) NOT NULL,
                        action VARCHAR(20) NOT NULL,
                        quantity INTEGER,
                        price DECIMAL(10,4),
                        side VARCHAR(10),
                        order_id VARCHAR(50),
                        signal_data JSONB,
                        portfolio_value DECIMAL(15,2),
                        mode VARCHAR(10) DEFAULT 'LIVE'
                    )
                """)
                
                # Portfolio snapshots table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_value DECIMAL(15,2),
                        cash_balance DECIMAL(15,2),
                        positions JSONB,
                        daily_pnl DECIMAL(10,2),
                        total_pnl DECIMAL(10,2)
                    )
                """)
                
                # Market data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        symbol VARCHAR(10) NOT NULL,
                        open_price DECIMAL(10,4),
                        high_price DECIMAL(10,4),
                        low_price DECIMAL(10,4),
                        close_price DECIMAL(10,4),
                        volume BIGINT,
                        sma50 DECIMAL(10,4),
                        sma200 DECIMAL(10,4),
                        rsi DECIMAL(5,2)
                    )
                """)
                
                # Signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        symbol VARCHAR(10) NOT NULL,
                        signal_type VARCHAR(20),
                        entry_price DECIMAL(10,4),
                        stop_loss DECIMAL(10,4),
                        take_profit DECIMAL(10,4),
                        confidence DECIMAL(3,2),
                        executed BOOLEAN DEFAULT FALSE,
                        signal_data JSONB
                    )
                """)
                
                self.connection.commit()
                logger.info("Database tables setup completed")
                
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def log_activity(self, symbol: str, action: str, **kwargs):
        """Log trading activity to database"""
        if not self.connection:
            if not self.connect():
                return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trading_activities 
                    (symbol, action, quantity, price, side, order_id, signal_data, portfolio_value, mode)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    action,
                    kwargs.get('quantity'),
                    kwargs.get('price'),
                    kwargs.get('side'),
                    kwargs.get('order_id'),
                    json.dumps(kwargs.get('signal_data', {})),
                    kwargs.get('portfolio_value'),
                    'DRYRUN' if DRY_RUN else 'LIVE'
                ))
                self.connection.commit()
                logger.debug(f"Activity logged: {action} {symbol}")
        except Exception as e:
            logger.error(f"Database logging error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def log_signal(self, signal: TradeSignal):
        """Log trading signal to database"""
        if not self.connection:
            if not self.connect():
                return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO signals 
                    (symbol, signal_type, entry_price, stop_loss, take_profit, confidence, signal_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    signal.symbol,
                    signal.signal_type,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.confidence,
                    json.dumps(asdict(signal))
                ))
                self.connection.commit()
        except Exception as e:
            logger.error(f"Signal logging error: {e}")
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio history for dashboard"""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM portfolio_snapshots 
                    WHERE timestamp >= %s 
                    ORDER BY timestamp DESC
                """, (datetime.utcnow() - timedelta(days=days),))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Portfolio history error: {e}")
            return []

class EnhancedBroker:
    """Enhanced multi-stock broker with improved features"""
    
    def __init__(self, key, secret, base_url):
        self.key = key
        self.secret = secret
        self.base = base_url
        self.client = None
        self.cached_data = {}
        self.last_cache_time = {}
        
        if tradeapi and key and secret:
            try:
                self.client = tradeapi.REST(key, secret, base_url, api_version="v2")
                account = self.client.get_account()
                logger.info(f"Alpaca client initialized. Account status: {account.status}")
            except Exception as e:
                logger.warning(f"Failed to init Alpaca client: {e}")
                self.client = None
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        if self.client:
            try:
                acc = self.client.get_account()
                return float(acc.equity)
            except Exception as e:
                logger.error(f"Failed to get account equity: {e}")
                return 100000.0
        return 100000.0
    
    def get_positions(self) -> Dict[str, PortfolioPosition]:
        """Get all current positions"""
        positions = {}
        
        if self.client:
            try:
                alpaca_positions = self.client.list_positions()
                for pos in alpaca_positions:
                    positions[pos.symbol] = PortfolioPosition(
                        symbol=pos.symbol,
                        quantity=int(pos.qty),
                        avg_price=float(pos.avg_cost),
                        current_price=float(pos.current_price or pos.avg_cost),
                        unrealized_pnl=float(pos.unrealized_pl or 0),
                        realized_pnl=0.0,  # Would need separate tracking
                        timestamp=datetime.utcnow()
                    )
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
        
        return positions
    
    def get_bars_multi(self, symbols: List[str], timeframe: str = "1D", limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols"""
        result = {}
        
        for symbol in symbols:
            try:
                result[symbol] = self.get_bars(symbol, timeframe, limit)
            except Exception as e:
                logger.error(f"Failed to get bars for {symbol}: {e}")
                continue
        
        return result
    
    def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 200) -> pd.DataFrame:
        """Get historical bars for a single symbol"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.cached_data and 
            current_time - self.last_cache_time.get(cache_key, 0) < 60):
            return self.cached_data[cache_key]
        
        if self.client:
            try:
                barset = self.client.get_bars(symbol, timeframe, limit=limit).df
                
                if barset is None or barset.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Handle multi-index DataFrames
                if isinstance(barset.index, pd.MultiIndex):
                    if symbol in barset.index.get_level_values(1):
                        df = barset.xs(symbol, level=1)
                    else:
                        raise ValueError(f"Symbol {symbol} not found in data")
                else:
                    df = barset
                
                # Ensure proper datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                df = df[~df.index.duplicated(keep='first')]
                
                # Validate columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")
                
                # Cache result
                self.cached_data[cache_key] = df
                self.last_cache_time[cache_key] = current_time
                
                return df
                
            except Exception as e:
                logger.error(f"Failed to get bars for {symbol}: {e}")
                if cache_key in self.cached_data:
                    return self.cached_data[cache_key]
                raise
        else:
            raise RuntimeError("Alpaca client unavailable")
    
    def submit_bracket_order(self, symbol: str, qty: int, side: str, 
                           stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> dict:
        """Submit bracket order"""
        if not self.client:
            logger.info(f"[SIM] Bracket order: {side} {qty} {symbol} | SL={stop_loss} TP={take_profit}")
            return {"sim": True, "symbol": symbol, "qty": qty, "side": side}
        
        try:
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "gtc"
            }
            
            if stop_loss and take_profit:
                order_params.update({
                    "order_class": "bracket",
                    "stop_loss": {"stop_price": f"{stop_loss:.2f}"},
                    "take_profit": {"limit_price": f"{take_profit:.2f}"}
                })
            
            order = self.client.submit_order(**order_params)
            logger.info(f"Placed bracket order: {order.id}")
            return {"success": True, "order_id": order.id, "order": order}
            
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {"error": str(e)}

# Technical Analysis Functions (similar to original but enhanced)
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average"""
    return series.rolling(window, min_periods=max(1, window//2)).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI calculation"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def find_recent_swing_high_low(df: pd.DataFrame, lookback: int = 30) -> dict:
    """Find recent swing high and low"""
    if len(df) < lookback + 2:
        lookback = max(5, len(df) - 2)
    
    window = df[-(lookback+1):-1] if len(df) > lookback else df[:-1]
    
    if window.empty:
        raise ValueError("No data for swing calculation")
    
    swing_high = window['high'].max()
    swing_low = window['low'].min()
    high_time = window['high'].idxmax()
    low_time = window['low'].idxmin()
    
    return {
        "high": float(swing_high),
        "high_time": high_time,
        "low": float(swing_low),
        "low_time": low_time,
        "lookback_used": lookback
    }

def compute_fib_levels(high: float, low: float) -> dict:
    """Compute Fibonacci retracement levels"""
    if high <= low:
        raise ValueError(f"Invalid swing points: high ({high}) <= low ({low})")
    
    diff = high - low
    levels = {
        "0.0": high,
        "23.6": high - 0.236 * diff,
        "38.2": high - 0.382 * diff,
        "50.0": high - 0.5 * diff,
        "61.8": high - 0.618 * diff,
        "78.6": high - 0.786 * diff,
        "100.0": low
    }
    return levels

class MultiStockAnalyzer:
    """Multi-stock technical analysis and signal generation"""
    
    def __init__(self, broker: EnhancedBroker, db: DatabaseManager, notifier: DiscordNotifier):
        self.broker = broker
        self.db = db
        self.notifier = notifier
        self.lookback = 30
        self.fib_tolerance = 0.006
    
    def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze a single symbol for trading signals"""
        try:
            df = self.broker.get_bars(symbol, limit=400)
            
            if len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate indicators
            swing = find_recent_swing_high_low(df, self.lookback)
            fibs = compute_fib_levels(swing["high"], swing["low"])
            price = float(df['close'].iloc[-1])
            
            sma50_val = sma(df['close'], 50).iloc[-1]
            sma200_val = sma(df['close'], 200).iloc[-1] if len(df) >= 200 else None
            rsi_val = rsi(df['close']).iloc[-1]
            
            # Check entry conditions
            tolerance = self.fib_tolerance
            near_50 = abs(price - fibs["50.0"]) / fibs["50.0"] <= tolerance
            near_618 = abs(price - fibs["61.8"]) / fibs["61.8"] <= tolerance
            
            sma_condition = price > sma50_val
            rsi_condition = rsi_val < 70
            fib_condition = near_50 or near_618
            trend_condition = True
            
            if sma200_val and not pd.isna(sma200_val):
                trend_condition = sma50_val > sma200_val
            
            # Calculate confidence score
            confidence = 0.0
            if fib_condition: confidence += 0.3
            if sma_condition: confidence += 0.25
            if rsi_condition: confidence += 0.2
            if trend_condition: confidence += 0.25
            
            # Additional scoring factors
            if near_618: confidence += 0.1  # 61.8% is stronger than 50%
            if rsi_val < 50: confidence += 0.1  # Not overbought
            
            confidence = min(1.0, confidence)
            
            if fib_condition and sma_condition and rsi_condition and trend_condition and confidence >= 0.7:
                # Calculate position sizing
                stop_price = swing["low"] * 0.98  # 2% below swing low
                equity = self.broker.get_account_equity()
                risk_amount = equity * RISK_PER_TRADE
                risk_per_share = price - stop_price
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                
                if quantity > 0:
                    take_profit = price + 1.5 * risk_per_share
                    
                    signal = TradeSignal(
                        symbol=symbol,
                        signal_type="long",
                        entry_price=price,
                        stop_loss=stop_price,
                        take_profit=take_profit,
                        quantity=quantity,
                        confidence=confidence,
                        fib_level="50.0" if near_50 else "61.8",
                        rsi=rsi_val,
                        sma50=sma50_val,
                        timestamp=datetime.utcnow()
                    )
                    
                    # Log signal to database
                    self.db.log_signal(signal)
                    
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def scan_watchlist(self) -> List[TradeSignal]:
        """Scan all symbols in watchlist for signals"""
        signals = []
        
        for symbol in WATCHLIST:
            try:
                signal = self.analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    
                    # Send Discord notification for new signal
                    fields = [
                        {"name": "Entry Price", "value": f"${signal.entry_price:.2f}", "inline": True},
                        {"name": "Stop Loss", "value": f"${signal.stop_loss:.2f}", "inline": True},
                        {"name": "Take Profit", "value": f"${signal.take_profit:.2f}", "inline": True},
                        {"name": "Quantity", "value": str(signal.quantity), "inline": True},
                        {"name": "Confidence", "value": f"{signal.confidence:.1%}", "inline": True},
                        {"name": "Fib Level", "value": signal.fib_level, "inline": True}
                    ]
                    
                    self.notifier.send_sync(
                        NotificationType.RECOMMENDATION,
                        f"🎯 Trading Signal: {symbol}",
                        f"New {signal.signal_type} signal detected for {symbol}",
                        fields
                    )
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return signals

class TradingBot:
    """Main enhanced trading bot with multi-stock support"""
    
    def __init__(self):
        self.broker = EnhancedBroker(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE)
        self.db = DatabaseManager()
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = MultiStockAnalyzer(self.broker, self.db, self.notifier)
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.start_time = datetime.utcnow()
    
    async def send_daily_update(self):
        """Send daily performance update"""
        try:
            equity = self.broker.get_account_equity()
            positions = self.broker.get_positions()
            
            total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
            
            fields = [
                {"name": "Account Equity", "value": f"${equity:,.2f}", "inline": True},
                {"name": "Active Positions", "value": str(len(positions)), "inline": True},
                {"name": "Unrealized P&L", "value": f"${total_unrealized:,.2f}", "inline": True},
                {"name": "Watchlist", "value": ", ".join(WATCHLIST[:5]), "inline": False}
            ]
            
            if positions:
                position_summary = "\n".join([
                    f"{pos.symbol}: {pos.quantity} @ ${pos.current_price:.2f} (${pos.unrealized_pnl:+.2f})"
                    for pos in list(positions.values())[:5]
                ])
                fields.append({"name": "Top Positions", "value": position_summary, "inline": False})
            
            await self.notifier.send_notification(
                NotificationType.DAILY_UPDATE,
                "📊 Daily Trading Update",
                f"Trading bot status as of {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                fields
            )
            
        except Exception as e:
            logger.error(f"Error sending daily update: {e}")
    
    async def send_service_status(self):
        """Send service status notification"""
        try:
            uptime = datetime.utcnow() - self.start_time
            
            fields = [
                {"name": "Status", "value": "🟢 Online", "inline": True},
                {"name": "Mode", "value": "DRY RUN" if DRY_RUN else "LIVE", "inline": True},
                {"name": "Uptime", "value": str(uptime).split('.')[0], "inline": True},
                {"name": "Watchlist Size", "value": str(len(WATCHLIST)), "inline": True},
                {"name": "Check Interval", "value": f"{CHECK_INTERVAL}s", "inline": True},
                {"name": "Max Positions", "value": str(MAX_TOTAL_POSITIONS), "inline": True}
            ]
            
            await self.notifier.send_notification(
                NotificationType.SERVICE_STATUS,
                "🤖 Bot Service Status",
                "Enhanced Trading Bot is running normally",
                fields
            )
            
        except Exception as e:
            logger.error(f"Error sending service status: {e}")
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Check if we can take more positions
            current_positions = len(self.broker.get_positions())
            if current_positions >= MAX_TOTAL_POSITIONS:
                logger.info(f"Max total positions reached: {current_positions}")
                return False
            
            # Check symbol-specific position limit
            existing_qty = self.broker.get_positions().get(signal.symbol)
            if existing_qty and abs(existing_qty.quantity) >= MAX_POSITIONS_PER_STOCK:
                logger.info(f"Max positions for {signal.symbol} reached")
                return False
            
            # Execute order
            result = self.broker.submit_bracket_order(
                signal.symbol,
                signal.quantity,
                "buy",
                signal.stop_loss,
                signal.take_profit
            )
            
            # Log the trade
            self.db.log_activity(
                signal.symbol,
                "BUY_SIGNAL_EXECUTED",
                quantity=signal.quantity,
                price=signal.entry_price,
                side="buy",
                order_id=result.get("order_id"),
                signal_data=asdict(signal),
                portfolio_value=self.broker.get_account_equity()
            )
            
            # Send Discord notification
            fields = [
                {"name": "Symbol", "value": signal.symbol, "inline": True},
                {"name": "Quantity", "value": str(signal.quantity), "inline": True},
                {"name": "Entry Price", "value": f"${signal.entry_price:.2f}", "inline": True},
                {"name": "Stop Loss", "value": f"${signal.stop_loss:.2f}", "inline": True},
                {"name": "Take Profit", "value": f"${signal.take_profit:.2f}", "inline": True},
                {"name": "Risk Amount", "value": f"${signal.quantity * (signal.entry_price - signal.stop_loss):.2f}", "inline": True}
            ]
            
            if result.get("sim"):
                self.notifier.send_sync(
                    NotificationType.PURCHASE,
                    f"🧪 Simulated Purchase: {signal.symbol}",
                    f"Would execute buy order for {signal.symbol}",
                    fields
                )
            else:
                self.notifier.send_sync(
                    NotificationType.PURCHASE,
                    f"💰 Purchase Executed: {signal.symbol}",
                    f"Successfully executed buy order for {signal.symbol}",
                    fields
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    async def run_main_loop(self):
        """Main trading loop"""
        global shutdown_flag
        
        consecutive_errors = 0
        last_daily_update = datetime.utcnow().date()
        last_status_update = datetime.utcnow()
        
        # Send startup notification
        await self.send_service_status()
        
        logger.info(f"Starting enhanced trading bot for {len(WATCHLIST)} symbols")
        
        while not shutdown_flag and consecutive_errors < 5:
            try:
                current_time = datetime.utcnow()
                
                # Daily updates
                if current_time.date() > last_daily_update:
                    await self.send_daily_update()
                    last_daily_update = current_time.date()
                
                # Hourly status updates
                if (current_time - last_status_update).seconds >= 3600:
                    await self.send_service_status()
                    last_status_update = current_time
                
                # Scan for signals
                signals = self.analyzer.scan_watchlist()
                
                # Execute signals
                for signal in signals:
                    if self.execute_signal(signal):
                        logger.info(f"Executed signal for {signal.symbol}")
                        # Add delay between orders
                        await asyncio.sleep(5)
                
                # Reset error counter
                consecutive_errors = 0
                
                logger.debug(f"Completed scan cycle, sleeping for {CHECK_INTERVAL} seconds")
                await asyncio.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                shutdown_flag = True
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in main loop (attempt {consecutive_errors}/5): {e}")
                
                if consecutive_errors >= 5:
                    await self.notifier.send_notification(
                        NotificationType.ERROR,
                        "🚨 Bot Error",
                        f"Trading bot stopped due to consecutive errors: {e}",
                        [{"name": "Error Count", "value": str(consecutive_errors), "inline": True}]
                    )
                    break
                
                await asyncio.sleep(min(CHECK_INTERVAL * consecutive_errors, 3600))
        
        # Cleanup
        await self.notifier.close_session()
        logger.info("Enhanced trading bot shutdown completed")

async def main():
    """Main entry point"""
    try:
        # Initialize bot
        bot = TradingBot()
        
        # Run health check
        logger.info("Starting Enhanced Trading Bot v3.0")
        logger.info(f"Watchlist: {', '.join(WATCHLIST)}")
        logger.info(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'}")
        
        if not DRY_RUN:
            confirmation = input("You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
            if confirmation != 'CONFIRM':
                logger.info("Live trading cancelled")
                return
        
        # Start main loop
        await bot.run_main_loop()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
