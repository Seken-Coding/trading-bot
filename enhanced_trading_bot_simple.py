#!/usr/bin/env python3
"""
Enhanced Multi-Stock Trading Bot v3.0

Features:
- Multi-stock support with configurable watchlist
- Discord webhook notifications
- PostgreSQL database logging
- Modern web dashboard
- Risk management and portfolio tracking

Usage:
    python enhanced_trading_bot_simple.py

Environment Variables:
    WATCHLIST=AMD,NVDA,AAPL,MSFT,GOOGL,TSLA
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    POSTGRES_HOST=localhost
    POSTGRES_DB=trading_bot
    POSTGRES_USER=trading_user
    POSTGRES_PASSWORD=your_password
    DRY_RUN=true
"""

import os
import time
import logging
import signal
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Configuration
WATCHLIST = os.getenv("WATCHLIST", "AMD,NVDA,AAPL,MSFT,GOOGL,TSLA").split(",")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

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

shutdown_flag = False

def signal_handler(signum, frame):
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class TradeSignal:
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    confidence: float
    timestamp: datetime

class DiscordNotifier:
    """Simple Discord webhook notifier"""
    
    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
    
    def send_notification(self, title: str, description: str, color: int = 0x00ff00):
        """Send Discord notification"""
        if not self.webhook_url:
            logger.debug("Discord webhook not configured")
            return
        
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Enhanced Trading Bot v3.0"}
        }
        
        payload = {
            "embeds": [embed],
            "username": "Trading Bot"
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                logger.debug(f"Discord notification sent: {title}")
            else:
                logger.warning(f"Discord notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Discord notification error: {e}")

class SimpleDatabase:
    """Simple database logger"""
    
    def __init__(self):
        self.connection = None
        self.setup_connection()
    
    def setup_connection(self):
        """Setup database connection if available"""
        if not HAS_POSTGRES:
            logger.info("PostgreSQL not available, logging to files only")
            return
        
        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            database = os.getenv("POSTGRES_DB", "trading_bot")
            user = os.getenv("POSTGRES_USER", "trading_user")
            password = os.getenv("POSTGRES_PASSWORD", "")
            
            if password:  # Only try to connect if password is provided
                self.connection = psycopg2.connect(
                    host=host, port=port, database=database,
                    user=user, password=password,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection established")
            else:
                logger.info("No database password provided, skipping DB connection")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
    
    def log_activity(self, activity_data: dict):
        """Log activity to database or file"""
        # Always log to file
        with open("trading_activities.log", "a") as f:
            f.write(f"{datetime.utcnow().isoformat()}: {json.dumps(activity_data)}\n")
        
        # Try database if available
        if self.connection:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO trading_activities 
                        (symbol, action, data, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        activity_data.get('symbol'),
                        activity_data.get('action'),
                        json.dumps(activity_data),
                        datetime.utcnow()
                    ))
                    self.connection.commit()
            except Exception as e:
                logger.error(f"Database logging failed: {e}")

class EnhancedBroker:
    """Enhanced broker with multi-stock support"""
    
    def __init__(self):
        self.client = None
        self.cached_data = {}
        self.last_cache_time = {}
        
        if tradeapi and ALPACA_KEY and ALPACA_SECRET:
            try:
                self.client = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version="v2")
                account = self.client.get_account()
                logger.info(f"Alpaca connected. Account status: {account.status}")
            except Exception as e:
                logger.warning(f"Alpaca connection failed: {e}")
        else:
            logger.info("Alpaca not configured, running in simulation mode")
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        if self.client:
            try:
                account = self.client.get_account()
                return float(account.equity)
            except Exception as e:
                logger.error(f"Failed to get equity: {e}")
        return 100000.0  # Default for simulation
    
    def get_bars(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Get historical bars for symbol"""
        cache_key = f"{symbol}_{limit}"
        current_time = time.time()
        
        # Check cache (1 minute TTL)
        if (cache_key in self.cached_data and 
            current_time - self.last_cache_time.get(cache_key, 0) < 60):
            return self.cached_data[cache_key]
        
        if self.client:
            try:
                bars = self.client.get_bars(symbol, "1Day", limit=limit).df
                
                if bars is None or bars.empty:
                    raise ValueError(f"No data for {symbol}")
                
                # Handle multi-index if needed
                if isinstance(bars.index, pd.MultiIndex) and symbol in bars.index.get_level_values(1):
                    df = bars.xs(symbol, level=1)
                else:
                    df = bars
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
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
            # Generate fake data for simulation
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
            np.random.seed(hash(symbol) % 2**32)  # Consistent fake data
            prices = 100 + np.cumsum(np.random.randn(limit) * 0.02)
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.randn(limit) * 0.001),
                'high': prices * (1 + np.abs(np.random.randn(limit)) * 0.01),
                'low': prices * (1 - np.abs(np.random.randn(limit)) * 0.01),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, limit)
            }, index=dates)
            
            return df
    
    def submit_order(self, symbol: str, qty: int, side: str) -> dict:
        """Submit market order"""
        if self.client and not DRY_RUN:
            try:
                order = self.client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Order submitted: {order.id}")
                return {"success": True, "order_id": str(order.id)}
            except Exception as e:
                logger.error(f"Order failed: {e}")
                return {"error": str(e)}
        else:
            logger.info(f"[SIM] Order: {side} {qty} {symbol}")
            return {"sim": True, "symbol": symbol, "qty": qty, "side": side}

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

def find_swing_levels(df: pd.DataFrame, lookback: int = 30) -> dict:
    """Find recent swing high and low"""
    if len(df) < lookback + 2:
        lookback = max(5, len(df) - 2)
    
    window = df[-(lookback+1):-1] if len(df) > lookback else df[:-1]
    
    return {
        "high": float(window['high'].max()),
        "low": float(window['low'].min())
    }

def compute_fib_levels(high: float, low: float) -> dict:
    """Compute Fibonacci retracement levels"""
    diff = high - low
    return {
        "50.0": high - 0.5 * diff,
        "61.8": high - 0.618 * diff
    }

class TradingAnalyzer:
    """Multi-stock technical analysis"""
    
    def __init__(self, broker: EnhancedBroker):
        self.broker = broker
        self.lookback = 30
        self.fib_tolerance = 0.006
    
    def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Analyze symbol for trading signals"""
        try:
            df = self.broker.get_bars(symbol, limit=200)
            
            if len(df) < 50:
                return None
            
            # Technical indicators
            swing = find_swing_levels(df, self.lookback)
            fibs = compute_fib_levels(swing["high"], swing["low"])
            price = float(df['close'].iloc[-1])
            
            sma50_val = sma(df['close'], 50).iloc[-1]
            rsi_val = rsi(df['close']).iloc[-1]
            
            # Entry conditions
            tolerance = self.fib_tolerance
            near_50 = abs(price - fibs["50.0"]) / fibs["50.0"] <= tolerance
            near_618 = abs(price - fibs["61.8"]) / fibs["61.8"] <= tolerance
            
            sma_condition = price > sma50_val
            rsi_condition = rsi_val < 70
            fib_condition = near_50 or near_618
            
            # Calculate confidence
            confidence = 0.0
            if fib_condition: confidence += 0.4
            if sma_condition: confidence += 0.3
            if rsi_condition: confidence += 0.3
            
            if fib_condition and sma_condition and rsi_condition and confidence >= 0.7:
                # Position sizing
                equity = self.broker.get_account_equity()
                stop_price = swing["low"] * 0.98
                risk_amount = equity * RISK_PER_TRADE
                risk_per_share = price - stop_price
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                
                if quantity > 0:
                    take_profit = price + 1.5 * risk_per_share
                    
                    return TradeSignal(
                        symbol=symbol,
                        signal_type="long",
                        entry_price=price,
                        stop_loss=stop_price,
                        take_profit=take_profit,
                        quantity=quantity,
                        confidence=confidence,
                        timestamp=datetime.utcnow()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def scan_watchlist(self) -> List[TradeSignal]:
        """Scan all symbols for signals"""
        signals = []
        
        for symbol in WATCHLIST:
            try:
                signal = self.analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal found for {symbol}: confidence {signal.confidence:.2%}")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return signals

class TradingBot:
    """Main trading bot"""
    
    def __init__(self):
        self.broker = EnhancedBroker()
        self.db = SimpleDatabase()
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TradingAnalyzer(self.broker)
        self.start_time = datetime.utcnow()
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trading signal"""
        try:
            result = self.broker.submit_order(signal.symbol, signal.quantity, "buy")
            
            # Log activity
            activity_data = {
                "symbol": signal.symbol,
                "action": "BUY_SIGNAL_EXECUTED",
                "quantity": signal.quantity,
                "price": signal.entry_price,
                "confidence": signal.confidence,
                "mode": "DRYRUN" if DRY_RUN else "LIVE"
            }
            
            self.db.log_activity(activity_data)
            
            # Discord notification
            title = f"{'🧪 Simulated' if DRY_RUN else '💰'} Purchase: {signal.symbol}"
            description = f"{'Would execute' if DRY_RUN else 'Executed'} buy order for {signal.symbol}"
            description += f"\nQuantity: {signal.quantity}"
            description += f"\nEntry: ${signal.entry_price:.2f}"
            description += f"\nStop: ${signal.stop_loss:.2f}"
            description += f"\nTarget: ${signal.take_profit:.2f}"
            description += f"\nConfidence: {signal.confidence:.1%}"
            
            self.notifier.send_notification(title, description)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    def send_daily_update(self):
        """Send daily portfolio update"""
        try:
            equity = self.broker.get_account_equity()
            uptime = datetime.utcnow() - self.start_time
            
            title = "📊 Daily Trading Update"
            description = f"Portfolio Value: ${equity:,.2f}"
            description += f"\nWatchlist: {', '.join(WATCHLIST[:3])}..."
            description += f"\nUptime: {str(uptime).split('.')[0]}"
            description += f"\nMode: {'DRY RUN' if DRY_RUN else 'LIVE'}"
            
            self.notifier.send_notification(title, description, color=0x0099ff)
            
        except Exception as e:
            logger.error(f"Error sending daily update: {e}")
    
    def run_main_loop(self):
        """Main trading loop"""
        global shutdown_flag
        
        consecutive_errors = 0
        last_daily_update = datetime.utcnow().date()
        
        # Send startup notification
        self.notifier.send_notification(
            "🤖 Bot Started",
            f"Enhanced Trading Bot v3.0 started\nWatchlist: {', '.join(WATCHLIST)}\nMode: {'DRY RUN' if DRY_RUN else 'LIVE'}",
            color=0x9900ff
        )
        
        logger.info(f"Starting trading bot for {len(WATCHLIST)} symbols")
        
        while not shutdown_flag and consecutive_errors < 5:
            try:
                current_time = datetime.utcnow()
                
                # Daily update
                if current_time.date() > last_daily_update:
                    self.send_daily_update()
                    last_daily_update = current_time.date()
                
                # Scan for signals
                signals = self.analyzer.scan_watchlist()
                
                # Execute signals
                for signal in signals:
                    if self.execute_signal(signal):
                        logger.info(f"Executed signal for {signal.symbol}")
                        time.sleep(5)  # Delay between orders
                
                if not signals:
                    logger.debug("No signals found in watchlist")
                
                consecutive_errors = 0
                
                logger.debug(f"Scan complete, sleeping {CHECK_INTERVAL} seconds")
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                shutdown_flag = True
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in main loop ({consecutive_errors}/5): {e}")
                
                if consecutive_errors >= 5:
                    self.notifier.send_notification(
                        "🚨 Bot Error",
                        f"Trading bot stopped due to errors: {e}",
                        color=0xff0000
                    )
                    break
                
                time.sleep(min(CHECK_INTERVAL * consecutive_errors, 3600))
        
        self.notifier.send_notification(
            "🛑 Bot Stopped",
            "Enhanced Trading Bot has been shut down",
            color=0xff6600
        )
        
        logger.info("Trading bot shutdown completed")

def main():
    """Main entry point"""
    logger.info("Enhanced Trading Bot v3.0 Starting")
    logger.info(f"Watchlist: {', '.join(WATCHLIST)}")
    logger.info(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'}")
    
    if not DRY_RUN:
        confirmation = input("Start LIVE trading? Type 'CONFIRM': ")
        if confirmation != 'CONFIRM':
            logger.info("Cancelled by user")
            return
    
    try:
        bot = TradingBot()
        bot.run_main_loop()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
