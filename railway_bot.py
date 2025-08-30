#!/usr/bin/env python3
"""
Railway-Optimized Enhanced Trading Bot

Lightweight version optimized for Railway deployment with:
- Minimal memory footprint
- Efficient database connections
- Optimized caching
- Resource-aware operation
"""

import os
import time
import logging
import signal
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Railway environment detection
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
MEMORY_LIMIT_MB = int(os.getenv('RAILWAY_MEMORY_LIMIT', '512'))  # Railway default

# Optimized configuration for Railway
WATCHLIST = os.getenv("WATCHLIST", "AMD,NVDA,AAPL,MSFT").split(",")  # Reduced default
MAX_SYMBOLS = 6  # Limit for memory efficiency
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "180"))  # More frequent
CACHE_TTL = 90  # Shorter cache for fresher data
BATCH_SIZE = 2  # Process symbols in smaller batches

# Database configuration for Railway
DATABASE_URL = os.getenv("DATABASE_URL")  # Railway provides this
POSTGRES_HOST = os.getenv("PGHOST", "localhost")
POSTGRES_PORT = int(os.getenv("PGPORT", "5432"))
POSTGRES_DB = os.getenv("PGDATABASE", "railway")
POSTGRES_USER = os.getenv("PGUSER", "postgres")
POSTGRES_PASSWORD = os.getenv("PGPASSWORD", "")

# Trading configuration
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# Optimized imports
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    tradeapi = None
    TimeFrame = None

try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

# Lightweight logging for Railway
class RailwayLogger:
    """Memory-efficient logger for Railway"""
    
    def __init__(self, name: str):
        self.name = name
        self.level = logging.INFO
        
        # Railway-optimized logging
        if IS_RAILWAY:
            # Railway captures stdout/stderr
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        else:
            # Local development with file
            handler = logging.FileHandler("trading_bot.log")
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        
        handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
    
    def info(self, msg: str): self.logger.info(msg)
    def error(self, msg: str): self.logger.error(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def debug(self, msg: str): self.logger.debug(msg)

logger = RailwayLogger("trading_bot")

# Global state
shutdown_flag = False

def signal_handler(signum, frame):
    global shutdown_flag
    logger.info(f"Shutdown signal {signum} received")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class TradeSignal:
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    confidence: float
    timestamp: datetime

class MemoryEfficientCache:
    """Memory-aware caching system"""
    
    def __init__(self, max_size: int = 50):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
    
    def get(self, key: str, ttl: int = CACHE_TTL):
        if key in self.cache:
            if time.time() - self.timestamps[key] < ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value):
        # Memory management
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
        gc.collect()  # Force garbage collection

class RailwayDatabase:
    """Lightweight database manager for Railway PostgreSQL"""
    
    def __init__(self):
        self.pool = None
        self.setup_connection()
    
    def setup_connection(self):
        """Setup connection pool for Railway"""
        if not HAS_POSTGRES:
            logger.warning("PostgreSQL not available")
            return
        
        try:
            # Use Railway's DATABASE_URL if available
            if DATABASE_URL:
                self.pool = SimpleConnectionPool(1, 3, DATABASE_URL)
            else:
                self.pool = SimpleConnectionPool(
                    1, 3,  # Min 1, Max 3 connections for efficiency
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD
                )
            logger.info("Database pool created")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        """Execute query with connection pooling"""
        if not self.pool:
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def log_activity(self, symbol: str, action: str, data: dict):
        """Log trading activity efficiently"""
        query = """
            INSERT INTO trading_activities (symbol, action, data, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        self.execute_query(query, (symbol, action, json.dumps(data), datetime.utcnow()))

class OptimizedBroker:
    """Memory-efficient broker for Railway"""
    
    def __init__(self):
        self.client = None
        self.cache = MemoryEfficientCache(max_size=20)  # Small cache
        self.last_equity_check = 0
        self.cached_equity = 100000.0
        
        if tradeapi and ALPACA_KEY and ALPACA_SECRET:
            try:
                self.client = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version="v2")
                logger.info("Alpaca client initialized")
            except Exception as e:
                logger.warning(f"Alpaca init failed: {e}")
    
    def get_account_equity(self) -> float:
        """Cached equity check to reduce API calls"""
        current_time = time.time()
        if current_time - self.last_equity_check < 300:  # 5-minute cache
            return self.cached_equity
        
        if self.client:
            try:
                account = self.client.get_account()
                self.cached_equity = float(account.equity)
                self.last_equity_check = current_time
            except Exception as e:
                logger.error(f"Equity check failed: {e}")
        
        return self.cached_equity
    
    def get_bars_batch(self, symbols: List[str], limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols efficiently"""
        results = {}
        
        # Process in smaller batches to manage memory
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i+BATCH_SIZE]
            
            for symbol in batch:
                cache_key = f"bars_{symbol}_{limit}"
                cached_data = self.cache.get(cache_key, ttl=CACHE_TTL)
                
                if cached_data is not None:
                    results[symbol] = cached_data
                    continue
                
                try:
                    if self.client:
                        # Use proper timeframe constant
                        timeframe = TimeFrame.Day if TimeFrame else "1Day"
                        bars = self.client.get_bars(symbol, timeframe, limit=limit).df
                        
                        # Handle different data structures
                        if isinstance(bars.index, pd.MultiIndex):
                            if symbol in bars.index.get_level_values(1):
                                df = bars.xs(symbol, level=1)
                            else:
                                continue
                        else:
                            df = bars
                        
                        # Optimize DataFrame
                        df = df[~df.index.duplicated(keep='first')]
                        df = df.astype({
                            'open': 'float32',
                            'high': 'float32', 
                            'low': 'float32',
                            'close': 'float32',
                            'volume': 'int32'
                        })
                        
                        self.cache.set(cache_key, df)
                        results[symbol] = df
                        
                    else:
                        # Generate lightweight fake data for testing
                        dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
                        np.random.seed(hash(symbol) % 2**32)
                        prices = 100 + np.cumsum(np.random.randn(limit) * 0.02)
                        
                        df = pd.DataFrame({
                            'open': prices.astype('float32'),
                            'high': (prices * 1.01).astype('float32'),
                            'low': (prices * 0.99).astype('float32'),
                            'close': prices.astype('float32'),
                            'volume': np.random.randint(1000000, 5000000, limit, dtype='int32')
                        }, index=dates)
                        
                        results[symbol] = df
                
                except Exception as e:
                    logger.error(f"Failed to get bars for {symbol}: {e}")
                    continue
                
                # Small delay to be API-friendly
                time.sleep(0.1)
            
            # Memory cleanup after each batch
            gc.collect()
        
        return results
    
    def submit_order(self, symbol: str, qty: int, side: str) -> dict:
        """Submit order with minimal overhead"""
        if self.client and not DRY_RUN:
            try:
                order = self.client.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type='market', time_in_force='gtc'
                )
                return {"success": True, "order_id": str(order.id)}
            except Exception as e:
                logger.error(f"Order failed: {e}")
                return {"error": str(e)}
        else:
            logger.info(f"[SIM] {side} {qty} {symbol}")
            return {"sim": True}

class LightweightAnalyzer:
    """Memory-efficient technical analysis"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> float:
        """Calculate SMA efficiently"""
        return series.tail(window).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> float:
        """Calculate RSI efficiently"""
        delta = series.diff().tail(period + 1)
        up = delta.clip(lower=0).mean()
        down = (-1 * delta.clip(upper=0)).mean()
        
        if down == 0:
            return 100
        
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
        """Find swing high/low efficiently"""
        window = df.tail(lookback)
        return float(window['high'].max()), float(window['low'].min())
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Lightweight technical analysis"""
        try:
            if len(df) < 50:
                return None
            
            # Get latest values efficiently
            close_prices = df['close']
            current_price = float(close_prices.iloc[-1])
            
            # Technical indicators
            sma50 = self.sma(close_prices, 50)
            rsi_val = self.rsi(close_prices, 14)
            swing_high, swing_low = self.find_swing_levels(df, 20)
            
            # Fibonacci levels
            fib_diff = swing_high - swing_low
            fib_50 = swing_high - 0.5 * fib_diff
            fib_618 = swing_high - 0.618 * fib_diff
            
            # Entry conditions (simplified for efficiency)
            tolerance = 0.006
            near_fib = (abs(current_price - fib_50) / fib_50 <= tolerance or 
                       abs(current_price - fib_618) / fib_618 <= tolerance)
            
            above_sma = current_price > sma50
            rsi_ok = rsi_val < 70
            
            # Calculate confidence
            confidence = 0.0
            if near_fib: confidence += 0.4
            if above_sma: confidence += 0.3
            if rsi_ok: confidence += 0.3
            
            if confidence >= 0.7:
                # Simple position sizing
                stop_loss = swing_low * 0.98
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = 100000 * RISK_PER_TRADE  # Simplified
                    quantity = int(risk_amount / risk_per_share)
                    
                    if quantity > 0:
                        return TradeSignal(
                            symbol=symbol,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=current_price + 1.5 * risk_per_share,
                            quantity=min(quantity, 100),  # Limit for safety
                            confidence=confidence,
                            timestamp=datetime.utcnow()
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None

class DiscordNotifier:
    """Lightweight Discord notifier"""
    
    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
    
    def send(self, title: str, description: str, color: int = 0x00ff00):
        """Send notification efficiently"""
        if not self.webhook_url:
            return
        
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
            logger.debug(f"Discord notification: {title}")
        except Exception as e:
            logger.error(f"Discord failed: {e}")

class RailwayTradingBot:
    """Main bot optimized for Railway deployment"""
    
    def __init__(self):
        self.broker = OptimizedBroker()
        self.db = RailwayDatabase()
        self.analyzer = LightweightAnalyzer()
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.start_time = datetime.utcnow()
        
        # Limit watchlist for memory efficiency
        self.watchlist = WATCHLIST[:MAX_SYMBOLS]
        
        logger.info(f"Bot initialized for {len(self.watchlist)} symbols")
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute signal efficiently"""
        try:
            result = self.broker.submit_order(signal.symbol, signal.quantity, "buy")
            
            # Log to database
            activity_data = {
                "symbol": signal.symbol,
                "quantity": signal.quantity,
                "price": signal.entry_price,
                "confidence": signal.confidence
            }
            
            self.db.log_activity(signal.symbol, "BUY_SIGNAL", activity_data)
            
            # Discord notification
            title = f"{'🧪 Test' if DRY_RUN else '💰'} Trade: {signal.symbol}"
            description = f"Entry: ${signal.entry_price:.2f}\nQty: {signal.quantity}\nConf: {signal.confidence:.1%}"
            
            self.notifier.send(title, description)
            
            return True
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            return False
    
    def run_scan_cycle(self):
        """Single scan cycle - memory efficient"""
        try:
            # Get market data in batch
            market_data = self.broker.get_bars_batch(self.watchlist, limit=100)
            
            signals_found = 0
            
            # Analyze each symbol
            for symbol, df in market_data.items():
                try:
                    signal = self.analyzer.analyze_symbol(symbol, df)
                    if signal:
                        if self.execute_signal(signal):
                            signals_found += 1
                            logger.info(f"Signal executed: {symbol} @ ${signal.entry_price:.2f}")
                except Exception as e:
                    logger.error(f"Symbol analysis failed {symbol}: {e}")
            
            logger.info(f"Scan complete: {signals_found} signals executed")
            
            # Memory cleanup
            del market_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Scan cycle failed: {e}")
    
    def send_status_update(self):
        """Send periodic status update"""
        try:
            uptime = datetime.utcnow() - self.start_time
            equity = self.broker.get_account_equity()
            
            title = "🤖 Bot Status"
            description = f"Uptime: {str(uptime).split('.')[0]}\n"
            description += f"Equity: ${equity:,.2f}\n"
            description += f"Watchlist: {len(self.watchlist)} symbols\n"
            description += f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}"
            
            self.notifier.send(title, description, color=0x0099ff)
            
        except Exception as e:
            logger.error(f"Status update failed: {e}")
    
    def run(self):
        """Main execution loop optimized for Railway"""
        global shutdown_flag
        
        logger.info("Railway Trading Bot starting...")
        
        # Send startup notification
        self.notifier.send(
            "🚀 Bot Started",
            f"Railway deployment active\nWatchlist: {', '.join(self.watchlist)}\nMemory limit: {MEMORY_LIMIT_MB}MB",
            color=0x9900ff
        )
        
        consecutive_errors = 0
        last_status_update = 0
        cycle_count = 0
        
        while not shutdown_flag and consecutive_errors < 3:
            try:
                cycle_start = time.time()
                
                # Run trading scan
                self.run_scan_cycle()
                
                # Periodic status update (every hour)
                current_time = time.time()
                if current_time - last_status_update > 3600:
                    self.send_status_update()
                    last_status_update = current_time
                
                # Performance monitoring
                cycle_duration = time.time() - cycle_start
                cycle_count += 1
                
                if cycle_count % 10 == 0:  # Every 10 cycles
                    logger.info(f"Cycle {cycle_count}: {cycle_duration:.2f}s")
                
                consecutive_errors = 0
                
                # Sleep until next cycle
                time.sleep(max(1, CHECK_INTERVAL - cycle_duration))
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt")
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error ({consecutive_errors}/3): {e}")
                
                if consecutive_errors >= 3:
                    self.notifier.send(
                        "🚨 Bot Error",
                        f"Bot stopped after {consecutive_errors} errors: {e}",
                        color=0xff0000
                    )
                    break
                
                time.sleep(min(60 * consecutive_errors, 300))  # Exponential backoff
        
        # Cleanup
        if hasattr(self.broker.cache, 'clear'):
            self.broker.cache.clear()
        
        self.notifier.send("🛑 Bot Stopped", "Railway bot shutdown", color=0xff6600)
        logger.info("Railway Trading Bot shutdown complete")

def main():
    """Main entry point optimized for Railway"""
    logger.info("Enhanced Trading Bot v3.0 - Railway Edition")
    logger.info(f"Environment: {'Railway' if IS_RAILWAY else 'Local'}")
    logger.info(f"Memory limit: {MEMORY_LIMIT_MB}MB")
    logger.info(f"Watchlist: {', '.join(WATCHLIST[:MAX_SYMBOLS])}")
    
    try:
        bot = RailwayTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
