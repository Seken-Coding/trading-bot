#!/usr/bin/env python3
"""
Enhanced Trading Bot - Clean Implementation

Multi-stock swing trading bot with:
- Discord webhook notifications
- PostgreSQL database logging  
- Web dashboard
- Railway cloud deployment support
- Clean, optimized code

Author: Enhanced Trading Bot System
Version: 4.0.0
"""

import os
import gc
import sys
import time
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

# Import handling for optional dependencies
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    tradeapi = None
    TimeFrame = None
    ALPACA_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    SimpleConnectionPool = None
    POSTGRES_AVAILABLE = False

try:
    import requests
    DISCORD_AVAILABLE = True
except ImportError:
    requests = None
    DISCORD_AVAILABLE = False

# Environment Detection
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
IS_PRODUCTION = os.getenv('ENVIRONMENT', 'development') == 'production'

# Configuration
WATCHLIST = os.getenv("WATCHLIST", "AMD,NVDA,AAPL,MSFT,GOOGL").split(",")
if IS_RAILWAY:
    WATCHLIST = WATCHLIST[:4]  # Limit for Railway

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET") 
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "600" if IS_RAILWAY else "300"))

# Trading parameters
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITIONS_PER_STOCK = int(os.getenv("MAX_POSITIONS_PER_STOCK", "1"))
TAKE_PROFIT_MULTIPLIER = float(os.getenv("TAKE_PROFIT_MULTIPLIER", "2.0"))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class DiscordNotifier:
    """Clean Discord webhook integration"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url and DISCORD_AVAILABLE
        if not self.enabled:
            logger.warning("Discord notifications disabled - webhook URL or requests library missing")
    
    def send(self, title: str, description: str, color: int = 0x00ff00) -> bool:
        """Send Discord notification"""
        if not self.enabled:
            return False
            
        try:
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            data = {"embeds": [embed]}
            response = requests.post(self.webhook_url, json=data, timeout=10)
            return response.status_code == 204
            
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False


class DatabaseManager:
    """Clean database connection handling"""
    
    def __init__(self):
        self.pool = None
        self.enabled = DATABASE_URL and POSTGRES_AVAILABLE
        
        if self.enabled:
            try:
                self.pool = SimpleConnectionPool(
                    1, 5 if IS_RAILWAY else 10,
                    DATABASE_URL
                )
                self._create_tables()
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                self.enabled = False
        else:
            logger.warning("Database disabled - URL or psycopg2 missing")
    
    def _create_tables(self):
        """Create required database tables"""
        query = """
        CREATE TABLE IF NOT EXISTS trading_activity (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol VARCHAR(10),
            action VARCHAR(50),
            quantity INTEGER,
            price DECIMAL(10,2),
            metadata JSONB
        );
        """
        self._execute_query(query)
    
    def _execute_query(self, query: str, params: Optional[tuple] = None):
        """Execute database query safely"""
        if not self.enabled:
            return None
            
        try:
            conn = self.pool.getconn()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Database query failed: {e}")
    
    def log_activity(self, symbol: str, action: str, **kwargs):
        """Log trading activity"""
        if not self.enabled:
            return
            
        query = """
        INSERT INTO trading_activity (symbol, action, quantity, price, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """
        params = (
            symbol,
            action,
            kwargs.get('quantity', 0),
            kwargs.get('price', 0),
            str(kwargs)
        )
        self._execute_query(query, params)


class TradingBroker:
    """Clean trading broker interface"""
    
    def __init__(self):
        self.client = None
        self.enabled = ALPACA_KEY and ALPACA_SECRET and ALPACA_AVAILABLE
        
        if self.enabled and not DRY_RUN:
            try:
                self.client = tradeapi.REST(
                    ALPACA_KEY,
                    ALPACA_SECRET,
                    ALPACA_BASE,
                    api_version="v2"
                )
                logger.info("Alpaca connection established")
            except Exception as e:
                logger.error(f"Alpaca connection failed: {e}")
                self.enabled = False
        else:
            logger.info("Broker in dry-run mode or disabled")
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        if not self.enabled or not self.client:
            return 100000.0  # Default for dry-run
            
        try:
            account = self.client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            return 100000.0
    
    def get_bars(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for symbol"""
        if not self.enabled or not self.client:
            # Return mock data for dry-run
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
            return pd.DataFrame({
                'open': np.random.uniform(100, 200, limit),
                'high': np.random.uniform(100, 200, limit),
                'low': np.random.uniform(100, 200, limit),
                'close': np.random.uniform(100, 200, limit),
                'volume': np.random.randint(1000000, 10000000, limit)
            }, index=dates)
        
        try:
            timeframe = TimeFrame.Day if TimeFrame else "1Day"
            bars = self.client.get_bars(symbol, timeframe, limit=limit).df
            return bars.tail(limit) if not bars.empty else None
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            return None
    
    def submit_order(self, symbol: str, quantity: int, side: str = "buy") -> Dict[str, Any]:
        """Submit trading order"""
        if DRY_RUN or not self.enabled or not self.client:
            return {
                "success": True,
                "order_id": f"dry_run_{symbol}_{int(time.time())}",
                "message": "Dry run order"
            }
        
        try:
            order = self.client.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            return {
                "success": True,
                "order_id": str(order.id) if order else "unknown",
                "message": "Order submitted"
            }
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return {"success": False, "error": str(e)}


class TechnicalAnalyzer:
    """Clean technical analysis implementation"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def find_swing_points(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, float]:
        """Find recent swing high and low"""
        recent_high = high.rolling(window=period).max().iloc[-1]
        recent_low = low.rolling(window=period).min().iloc[-1]
        return {"swing_high": recent_high, "swing_low": recent_low}
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze symbol for trading signals"""
        if df is None or df.empty or len(df) < 50:
            return None
        
        try:
            # Calculate indicators
            df['sma_50'] = self.calculate_sma(df['close'], 50)
            df['sma_200'] = self.calculate_sma(df['close'], 200)
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # Current values
            current_price = df['close'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            sma_200 = df['sma_200'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Swing points
            swing_points = self.find_swing_points(df['high'], df['low'])
            
            # Calculate Fibonacci levels
            swing_high = swing_points['swing_high']
            swing_low = swing_points['swing_low']
            fib_618 = swing_high - (swing_high - swing_low) * 0.618
            fib_50 = swing_high - (swing_high - swing_low) * 0.5
            
            # Signal conditions
            bullish_trend = current_price > sma_50 > sma_200
            oversold_rsi = 30 < rsi < 50
            at_fib_level = abs(current_price - fib_618) / current_price < 0.01 or \
                          abs(current_price - fib_50) / current_price < 0.01
            
            if bullish_trend and oversold_rsi and at_fib_level:
                return {
                    "symbol": symbol,
                    "signal": "buy",
                    "entry_price": current_price,
                    "stop_loss": swing_low * 0.98,  # 2% below swing low
                    "take_profit": current_price + (current_price - swing_low) * TAKE_PROFIT_MULTIPLIER,
                    "confidence": 0.75,
                    "rsi": rsi,
                    "reason": f"Bullish trend + RSI {rsi:.1f} + Fib retracement"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None


class TradingBot:
    """Main trading bot implementation"""
    
    def __init__(self):
        self.broker = TradingBroker()
        self.db = DatabaseManager()
        self.notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.analyzer = TechnicalAnalyzer()
        self.start_time = datetime.utcnow()
        
        logger.info(f"Trading Bot initialized (Railway: {IS_RAILWAY}, DryRun: {DRY_RUN})")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        account_equity = self.broker.get_account_equity()
        risk_amount = account_equity * RISK_PER_TRADE
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(risk_amount / risk_per_share)
        return max(1, min(quantity, 100))  # Limit to reasonable range
    
    def execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Execute trading signal"""
        try:
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, entry_price, stop_loss)
            if quantity <= 0:
                logger.warning(f"Invalid quantity calculated for {symbol}")
                return False
            
            # Submit order
            result = self.broker.submit_order(symbol, quantity, "buy")
            
            # Log activity
            self.db.log_activity(
                symbol=symbol,
                action="BUY_SIGNAL",
                quantity=quantity,
                price=entry_price,
                confidence=signal.get('confidence', 0),
                stop_loss=stop_loss,
                take_profit=signal.get('take_profit', 0)
            )
            
            # Send notification
            title = f"{'🧪 Simulated' if DRY_RUN else '💰'} Purchase: {symbol}"
            description = f"{'Would execute' if DRY_RUN else 'Executed'} buy order\n"
            description += f"Quantity: {quantity}\n"
            description += f"Entry: ${entry_price:.2f}\n"
            description += f"Stop: ${stop_loss:.2f}\n"
            description += f"Confidence: {signal.get('confidence', 0):.1%}"
            
            self.notifier.send(title, description, color=0x00ff00)
            
            logger.info(f"Signal executed for {symbol}: {quantity} shares @ ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def scan_watchlist(self) -> List[Dict[str, Any]]:
        """Scan watchlist for trading signals"""
        signals = []
        
        for symbol in WATCHLIST:
            try:
                df = self.broker.get_bars(symbol)
                signal = self.analyzer.analyze_symbol(symbol, df)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return signals
    
    def send_daily_update(self):
        """Send daily portfolio update"""
        try:
            equity = self.broker.get_account_equity()
            uptime = datetime.utcnow() - self.start_time
            
            title = "📊 Daily Trading Update"
            description = f"Portfolio Value: ${equity:,.2f}\n"
            description += f"Watchlist: {', '.join(WATCHLIST)}\n"
            description += f"Uptime: {str(uptime).split('.')[0]}\n"
            description += f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}"
            
            self.notifier.send(title, description, color=0x0099ff)
            
        except Exception as e:
            logger.error(f"Error sending daily update: {e}")
    
    def run(self):
        """Main trading loop"""
        global shutdown_flag
        
        logger.info("Enhanced Trading Bot starting...")
        logger.info(f"Watchlist: {', '.join(WATCHLIST)}")
        logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
        
        # Send startup notification
        self.notifier.send(
            "🚀 Bot Started",
            f"Enhanced Trading Bot v4.0\nWatchlist: {', '.join(WATCHLIST)}\nMode: {'DRY RUN' if DRY_RUN else 'LIVE'}",
            color=0x9900ff
        )
        
        consecutive_errors = 0
        last_daily_update = datetime.utcnow().date()
        
        while not shutdown_flag and consecutive_errors < 5:
            try:
                cycle_start = time.time()
                
                # Daily update check
                current_date = datetime.utcnow().date()
                if current_date > last_daily_update:
                    self.send_daily_update()
                    last_daily_update = current_date
                
                # Scan for signals
                signals = self.scan_watchlist()
                logger.info(f"Found {len(signals)} trading signals")
                
                # Execute signals
                for signal in signals:
                    if self.execute_signal(signal):
                        time.sleep(5)  # Delay between orders
                
                # Memory cleanup for Railway
                if IS_RAILWAY:
                    gc.collect()
                
                consecutive_errors = 0
                cycle_duration = time.time() - cycle_start
                
                logger.debug(f"Scan cycle completed in {cycle_duration:.2f}s")
                time.sleep(max(1, CHECK_INTERVAL - cycle_duration))
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in main loop ({consecutive_errors}/5): {e}")
                
                if consecutive_errors >= 5:
                    self.notifier.send(
                        "🚨 Bot Error",
                        f"Trading bot stopped due to errors: {e}",
                        color=0xff0000
                    )
                    break
                
                time.sleep(min(CHECK_INTERVAL * consecutive_errors, 3600))
        
        # Shutdown notification
        self.notifier.send(
            "🛑 Bot Stopped",
            "Enhanced Trading Bot has been shut down",
            color=0xff6600
        )
        
        logger.info("Trading bot shutdown completed")


def main():
    """Main entry point"""
    logger.info("Enhanced Trading Bot v4.0 Starting")
    logger.info(f"Environment: {'Railway' if IS_RAILWAY else 'Local'}")
    logger.info(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'}")
    
    if not DRY_RUN and not IS_RAILWAY:
        confirmation = input("Start LIVE trading? Type 'CONFIRM': ")
        if confirmation != 'CONFIRM':
            logger.info("Cancelled by user")
            return
    
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
