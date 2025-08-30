"""
Enhanced AMD Swing Trader (Alpaca paper by default)

Features:
- Fetch recent bars for SYMBOL (default AMD)
- Compute SMA50, SMA200, RSI, recent swing high/low and Fibonacci retracement levels
- Signal entry when price pulls back to 50% or 61.8% fib AND price > SMA50 (configurable)
- Position sizing: risk-based (e.g., risk 1% account equity per trade)
- Dry-run/paper by default; modular to enable live trading
- Places bracket orders (take-profit & stop-loss) if live mode enabled and Alpaca keys provided
- Enhanced error handling, position management, and safety features

Usage on Railway:
- Add env vars (ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)
- Set DRY_RUN=true while testing
"""

import os
import time
import math
import logging
import signal
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import json

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None  # handle missing package gracefully for dry-run

# CONFIG (can be overridden with env vars)
SYMBOL = os.getenv("SYMBOL", "AMD")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # fraction of equity to risk
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))  # polling interval
MAX_POSITION_RISK_PCT = float(os.getenv("MAX_POSITION_RISK_PCT", "0.02"))  # maximum fraction of equity
TAKE_PROFIT_MULTIPLIER = float(os.getenv("TP_MULT", "1.5"))  # e.g., 1.5x risk
STOP_LOSS_BUFFER = float(os.getenv("SL_BUFFER", "0.002"))  # small buffer below swing low
MAX_CONSECUTIVE_ERRORS = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "5"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))  # 5% daily loss limit
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "1"))
SIGNAL_VALIDATION_DRIFT = float(os.getenv("SIGNAL_DRIFT_THRESHOLD", "0.01"))  # 1%

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("amd_swing_trader")

# Global shutdown flag
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class TradingConfig:
    """Configuration management with validation"""
    
    def __init__(self):
        self.lookback = max(10, int(os.getenv("LOOKBACK", "30")))
        self.fib_tolerance = max(0.001, min(0.05, float(os.getenv("FIB_TOL", "0.006"))))
        self.require_sma50 = os.getenv("REQUIRE_SMA50", "true").lower() == "true"
        self.rsi_overbought = max(50, min(90, float(os.getenv("RSI_OVERBOUGHT", "70"))))
        self.min_volume_ratio = float(os.getenv("MIN_VOLUME_RATIO", "0.5"))
        self.cache_duration = int(os.getenv("CACHE_DURATION", "60"))  # seconds
        
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < RISK_PER_TRADE <= 0.1, "RISK_PER_TRADE must be between 0 and 0.1"
        assert 0 < MAX_POSITION_RISK_PCT <= 0.5, "MAX_POSITION_RISK_PCT must be between 0 and 0.5"
        assert TAKE_PROFIT_MULTIPLIER > 0, "TAKE_PROFIT_MULTIPLIER must be positive"
        assert CHECK_INTERVAL >= 30, "CHECK_INTERVAL must be at least 30 seconds"
        logger.info("Configuration validated successfully")

class PerformanceTracker:
    """Track trading performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = {}
        self.initial_equity = None
        
    def record_trade(self, trade_data):
        trade_data['timestamp'] = datetime.utcnow().isoformat()
        self.trades.append(trade_data)
        
    def get_daily_pnl(self, date=None):
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        return self.daily_pnl.get(date, 0.0)
        
    def update_daily_pnl(self, pnl, date=None):
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        self.daily_pnl[date] = self.daily_pnl.get(date, 0.0) + pnl
        
    def check_daily_loss_limit(self, current_equity):
        if self.initial_equity is None:
            self.initial_equity = current_equity
            
        today_pnl = self.get_daily_pnl()
        daily_loss_pct = abs(today_pnl) / self.initial_equity if self.initial_equity > 0 else 0
        
        if today_pnl < 0 and daily_loss_pct >= DAILY_LOSS_LIMIT:
            logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2%} >= {DAILY_LOSS_LIMIT:.2%}")
            return False
        return True

class EnhancedBroker:
    """Enhanced Alpaca wrapper with improved error handling and features"""
    
    def __init__(self, key, secret, base_url):
        self.key = key
        self.secret = secret
        self.base = base_url
        self.client = None
        self.last_cache_time = {}
        self.cached_data = {}
        
        if tradeapi and key and secret:
            try:
                self.client = tradeapi.REST(key, secret, base_url, api_version="v2")
                # Test connection
                account = self.client.get_account()
                logger.info(f"Alpaca client initialized. Account status: {account.status}")
            except Exception as e:
                logger.warning(f"Failed to init Alpaca client: {e}")
                self.client = None
        else:
            logger.info("Alpaca credentials not provided or alpaca-trade-api not installed")

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if self.client:
            try:
                clock = self.client.get_clock()
                return clock.is_open
            except Exception as e:
                logger.warning(f"Failed to check market status: {e}")
                return False
        else:
            # Simple heuristic for demo mode
            now = datetime.now()
            return now.weekday() < 5 and 9 <= now.hour < 16

    def get_account_equity(self) -> float:
        """Get current account equity with error handling"""
        if self.client:
            try:
                acc = self.client.get_account()
                return float(acc.equity)
            except Exception as e:
                logger.error(f"Failed to get account equity: {e}")
                return 100000.0  # fallback
        else:
            return 100000.0  # default simulated equity

    def get_existing_position(self, symbol: str) -> float:
        """Check for existing position in symbol"""
        if self.client:
            try:
                position = self.client.get_position(symbol)
                return float(position.qty)
            except Exception as e:
                if "position does not exist" not in str(e).lower():
                    logger.warning(f"Error checking position for {symbol}: {e}")
                return 0.0
        return 0.0

    def get_open_orders(self, symbol: str = None) -> list:
        """Get list of open orders"""
        if self.client:
            try:
                orders = self.client.list_orders(status='open', symbols=symbol)
                return orders
            except Exception as e:
                logger.error(f"Failed to get open orders: {e}")
                return []
        return []

    def get_bars(self, symbol: str, timeframe: str = "1D", limit: int = 200) -> pd.DataFrame:
        """Get historical bars with improved error handling and caching"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.cached_data and 
            current_time - self.last_cache_time.get(cache_key, 0) < 60):
            logger.debug(f"Returning cached data for {cache_key}")
            return self.cached_data[cache_key]
            
        if self.client:
            try:
                # Get bars from Alpaca
                barset = self.client.get_bars(symbol, timeframe, limit=limit).df
                
                if barset is None or barset.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Handle both multi-index and single-index DataFrames
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
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
                # Validate essential columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Cache the result
                self.cached_data[cache_key] = df
                self.last_cache_time[cache_key] = current_time
                
                logger.debug(f"Retrieved {len(df)} bars for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Failed to get bars for {symbol}: {e}")
                # Try to return cached data if available
                if cache_key in self.cached_data:
                    logger.info("Returning stale cached data due to API error")
                    return self.cached_data[cache_key]
                raise
        else:
            raise RuntimeError("Alpaca client unavailable for get_bars")

    def submit_bracket_order(self, symbol: str, qty: int, side: str, 
                           type: str = "market", stop_loss: float = None, 
                           take_profit: float = None) -> dict:
        """Submit bracket order with enhanced error handling"""
        if not self.client:
            logger.info(f"[SIM] Bracket order: {side} {qty} {symbol} | SL={stop_loss} TP={take_profit}")
            return {"sim": True, "symbol": symbol, "qty": qty, "side": side}

        try:
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": type,
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

# Enhanced indicator helpers
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with validation"""
    if len(series) < window:
        logger.warning(f"Insufficient data for SMA{window}: {len(series)} < {window}")
    return series.rolling(window, min_periods=max(1, window//2)).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using traditional simple moving average method"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Use simple moving average instead of exponential
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def find_recent_swing_high_low(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Find the most recent local swing high and low within lookback days.
    Enhanced with validation and context.
    """
    if len(df) < lookback + 2:
        lookback = max(5, len(df) - 2)
        logger.warning(f"Adjusted lookback to {lookback} due to insufficient data")
    
    if lookback <= 0:
        raise ValueError("Insufficient data for swing calculation")
        
    # Exclude the latest candle to avoid intraday noise
    window = df[-(lookback+1):-1] if len(df) > lookback else df[:-1]
    
    if window.empty:
        raise ValueError("No data available for swing calculation")
    
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

def get_latest_price(df: pd.DataFrame) -> float:
    """Get latest price with validation"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    return float(df['close'].iloc[-1])

def validate_signal(df: pd.DataFrame, signal: dict, config: TradingConfig) -> bool:
    """Enhanced signal validation"""
    try:
        current_price = get_latest_price(df)
        
        # Check if price hasn't moved too far from signal price
        price_drift = abs(current_price - signal["entry_price"]) / signal["entry_price"]
        if price_drift > SIGNAL_VALIDATION_DRIFT:
            logger.info(f"Signal invalidated by price drift: {price_drift:.2%}")
            return False
        
        # Check volume confirmation
        if len(df) >= 20:
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < config.min_volume_ratio:
                logger.info(f"Signal invalidated by low volume: {volume_ratio:.2f}")
                return False
        
        # Check that we're still near fibonacci levels
        tolerance = config.fib_tolerance
        fib_50 = signal.get("fib_50", signal.get("fib_level"))
        fib_618 = signal.get("fib_618", signal.get("fib_level"))
        
        near_fib = False
        if fib_50:
            near_fib = near_fib or abs(current_price - fib_50) / fib_50 <= tolerance
        if fib_618:
            near_fib = near_fib or abs(current_price - fib_618) / fib_618 <= tolerance
            
        if not near_fib:
            logger.info("Signal invalidated: price moved away from fibonacci levels")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating signal: {e}")
        return False

def check_entry_conditions(df: pd.DataFrame, config: TradingConfig) -> dict:
    """
    Enhanced entry condition checking with better validation
    """
    try:
        if len(df) < 50:
            logger.warning("Insufficient data for analysis")
            return {"signal": None, "reason": "insufficient_data"}
        
        swing = find_recent_swing_high_low(df, lookback=config.lookback)
        fibs = compute_fib_levels(swing["high"], swing["low"])
        price = get_latest_price(df)
        
        # Calculate indicators
        sma50 = sma(df['close'], 50).iloc[-1]
        sma200 = sma(df['close'], 200).iloc[-1] if len(df) >= 200 else None
        current_rsi = rsi(df['close']).iloc[-1]
        
        # Validate indicators
        if pd.isna(sma50) or pd.isna(current_rsi):
            logger.warning("Invalid indicator values calculated")
            return {"signal": None, "reason": "invalid_indicators"}
        
        tolerance = config.fib_tolerance

        # Check proximity to fib levels
        def near_level(level_price):
            return abs(price - level_price) / level_price <= tolerance

        near_50 = near_level(fibs["50.0"])
        near_618 = near_level(fibs["61.8"])
        
        # Entry conditions
        sma_condition = not config.require_sma50 or price > sma50
        rsi_condition = current_rsi < config.rsi_overbought
        fib_condition = near_50 or near_618
        
        # Additional trend confirmation
        trend_condition = True
        if sma200 is not None and not pd.isna(sma200):
            trend_condition = sma50 > sma200  # uptrend confirmation
        
        logger.debug(f"Entry conditions - Price: {price:.2f}, Fib50: {fibs['50.0']:.2f}, "
                    f"Fib618: {fibs['61.8']:.2f}, SMA50: {sma50:.2f}, RSI: {current_rsi:.1f}")
        logger.debug(f"Conditions - Fib: {fib_condition}, SMA: {sma_condition}, "
                    f"RSI: {rsi_condition}, Trend: {trend_condition}")
        
        if fib_condition and sma_condition and rsi_condition and trend_condition:
            level_hit = "50.0" if near_50 else "61.8"
            return {
                "signal": "long",
                "entry_price": price,
                "swing_high": swing["high"],
                "swing_low": swing["low"],
                "fib_level": float(fibs[level_hit]),
                "fib_50": float(fibs["50.0"]),
                "fib_618": float(fibs["61.8"]),
                "level_label": level_hit,
                "sma50": float(sma50),
                "sma200": float(sma200) if sma200 is not None else None,
                "rsi": float(current_rsi),
                "swing_data": swing
            }
        else:
            reasons = []
            if not fib_condition: reasons.append("not_near_fib")
            if not sma_condition: reasons.append("below_sma50")
            if not rsi_condition: reasons.append("rsi_overbought")
            if not trend_condition: reasons.append("downtrend")
            
            return {"signal": None, "reason": "_".join(reasons)}
            
    except Exception as e:
        logger.error(f"Error checking entry conditions: {e}")
        return {"signal": None, "reason": f"error: {str(e)}"}

def calc_position_size(account_equity: float, entry_price: float, stop_price: float, 
                      risk_per_trade: float = RISK_PER_TRADE, 
                      max_alloc_pct: float = MAX_POSITION_RISK_PCT) -> int:
    """
    Enhanced position sizing with additional safety checks
    """
    try:
        if entry_price <= 0 or stop_price <= 0 or account_equity <= 0:
            logger.error("Invalid inputs for position sizing")
            return 0
            
        if entry_price <= stop_price:
            logger.error(f"Invalid order: entry ({entry_price}) <= stop ({stop_price}) for long position")
            return 0
        
        risk_amount = account_equity * risk_per_trade
        per_share_risk = entry_price - stop_price
        qty_by_risk = math.floor(risk_amount / per_share_risk)
        
        # Cap by maximum allocation
        max_notional = account_equity * max_alloc_pct
        qty_by_allocation = math.floor(max_notional / entry_price)
        
        # Take the smaller of the two
        qty = min(qty_by_risk, qty_by_allocation)
        
        logger.debug(f"Position sizing - Risk qty: {qty_by_risk}, Allocation qty: {qty_by_allocation}, Final: {qty}")
        
        return max(0, int(qty))
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

def calculate_stop_loss(signal: dict, buffer: float = STOP_LOSS_BUFFER) -> float:
    """Calculate stop loss price with proper logic for long positions"""
    swing_low = signal["swing_low"]
    entry_price = signal["entry_price"]
    
    # Stop should be below swing low but not too far below entry
    stop_by_swing = swing_low * (1 - buffer)
    stop_by_entry = entry_price * 0.98  # 2% below entry as maximum stop
    
    # For long positions, use the higher of the two (closer to entry)
    stop_price = max(stop_by_swing, stop_by_entry)
    
    # Ensure stop is below entry price
    if stop_price >= entry_price:
        stop_price = entry_price * 0.99
        logger.warning(f"Adjusted stop loss to {stop_price:.2f} (was >= entry price)")
    
    return stop_price

def record_trade_journal(row: dict):
    """Enhanced trade journal recording"""
    try:
        journal_path = os.getenv("TRADE_JOURNAL", "trade_journal.csv")
        df = pd.DataFrame([row])
        header = not os.path.exists(journal_path)
        df.to_csv(journal_path, mode="a", index=False, header=header)
        logger.info("Trade recorded in journal")
    except Exception as e:
        logger.error(f"Failed to record trade: {e}")

def main_loop():
    """Enhanced main trading loop with comprehensive error handling"""
    global shutdown_flag
    
    try:
        # Initialize components
        config = TradingConfig()
        config.validate()
        
        broker = EnhancedBroker(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE)
        performance = PerformanceTracker()
        
        consecutive_errors = 0
        last_signal_time = None
        
        logger.info(f"Starting trading loop for {SYMBOL}")
        logger.info(f"Configuration: lookback={config.lookback}, fib_tolerance={config.fib_tolerance:.3f}")
        
        while not shutdown_flag and consecutive_errors < MAX_CONSECUTIVE_ERRORS:
            try:
                # Check if market is open
                if not broker.is_market_open():
                    logger.debug("Market is closed, waiting...")
                    time.sleep(min(CHECK_INTERVAL, 300))  # Check every 5 minutes when closed
                    continue
                
                # Check daily loss limit
                current_equity = broker.get_account_equity()
                if not performance.check_daily_loss_limit(current_equity):
                    logger.warning("Daily loss limit reached, stopping trading for today")
                    time.sleep(3600)  # Wait 1 hour before checking again
                    continue
                
                # Check existing positions
                existing_position = broker.get_existing_position(SYMBOL)
                if abs(existing_position) >= MAX_POSITIONS:
                    logger.debug(f"Maximum positions reached for {SYMBOL}: {existing_position}")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Check for open orders
                open_orders = broker.get_open_orders(SYMBOL)
                if len(open_orders) > 0:
                    logger.debug(f"Existing open orders for {SYMBOL}, skipping new signals")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Get market data
                df = broker.get_bars(SYMBOL, timeframe="1D", limit=400)
                
                # Check entry conditions
                signal = check_entry_conditions(df, config)
                
                if signal["signal"] == "long":
                    # Avoid duplicate signals
                    current_time = time.time()
                    if last_signal_time and current_time - last_signal_time < CHECK_INTERVAL:
                        logger.debug("Skipping duplicate signal")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    
                    # Validate signal
                    if not validate_signal(df, signal, config):
                        logger.info("Signal validation failed")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    
                    logger.info(f"Entry signal detected for {SYMBOL}: {signal['level_label']} fib "
                              f"at {signal['fib_level']:.2f}, price={signal['entry_price']:.2f}, "
                              f"rsi={signal['rsi']:.1f}")
                    
                    # Calculate position sizing
                    stop_price = calculate_stop_loss(signal)
                    qty = calc_position_size(current_equity, signal["entry_price"], stop_price)
                    
                    if qty <= 0:
                        logger.warning("Calculated quantity is 0, insufficient risk budget")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    
                    # Calculate take profit
                    risk_per_share = signal["entry_price"] - stop_price
                    take_profit_price = signal["entry_price"] + TAKE_PROFIT_MULTIPLIER * risk_per_share
                    risk_amount = risk_per_share * qty
                    
                    logger.info(f"Position details: equity=${current_equity:.2f}, qty={qty}, "
                              f"entry=${signal['entry_price']:.2f}, stop=${stop_price:.2f}, "
                              f"tp=${take_profit_price:.2f}, risk=${risk_amount:.2f}")
                    
                    # Record trade details
                    trade_record = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": SYMBOL,
                        "signal_type": "entry",
                        "side": "buy",
                        "qty": qty,
                        "entry_price": signal["entry_price"],
                        "stop_loss": stop_price,
                        "take_profit": take_profit_price,
                        "risk_amount": risk_amount,
                        "fib_level": signal["level_label"],
                        "rsi": signal["rsi"],
                        "sma50": signal["sma50"],
                        "account_equity": current_equity,
                        "mode": "DRYRUN" if DRY_RUN else "LIVE"
                    }
                    
                    if DRY_RUN or not broker.client:
                        # Simulate the trade
                        logger.info(f"[DRYRUN] Would place order: BUY {qty} {SYMBOL} at {signal['entry_price']:.2f} "
                                  f"| SL {stop_price:.2f} | TP {take_profit_price:.2f}")
                        performance.record_trade(trade_record)
                    else:
                        # Place live bracket order
                        order_result = broker.submit_bracket_order(
                            SYMBOL, qty, "buy",
                            type="market",
                            stop_loss=stop_price,
                            take_profit=take_profit_price
                        )
                        
                        trade_record.update({
                            "order_result": str(order_result),
                            "order_id": order_result.get("order_id")
                        })
                        performance.record_trade(trade_record)
                        
                        if "error" in order_result:
                            logger.error(f"Order placement failed: {order_result['error']}")
                        else:
                            logger.info(f"Successfully placed bracket order: {order_result.get('order_id')}")
                    
                    # Record in journal
                    record_trade_journal(trade_record)
                    last_signal_time = current_time
                    
                else:
                    reason = signal.get("reason", "unknown")
                    logger.debug(f"No entry signal for {SYMBOL}: {reason}")
                
                # Reset error counter on successful iteration
                consecutive_errors = 0
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                shutdown_flag = True
                break
                
            except Exception as e:
                consecutive_errors += 1
                error_sleep = min(CHECK_INTERVAL * consecutive_errors, 3600)  # Max 1 hour
                
                logger.error(f"Error in main loop (attempt {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical("Maximum consecutive errors reached, shutting down")
                    break
                
                logger.info(f"Waiting {error_sleep} seconds before retry...")
                time.sleep(error_sleep)
                continue
            
            if not shutdown_flag:
                logger.debug(f"Sleeping for {CHECK_INTERVAL} seconds...")
                time.sleep(CHECK_INTERVAL)
        
        # Cleanup
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.critical("Bot stopped due to excessive errors")
        else:
            logger.info("Bot shutdown completed successfully")
            
    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}")
        sys.exit(1)

def run_health_check():
    """Perform system health check before starting"""
    logger.info("Performing health check...")
    
    # Check required environment variables
    if not DRY_RUN:
        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.error("Alpaca credentials required for live trading")
            return False
    
    # Test broker connection
    try:
        broker = EnhancedBroker(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE)
        equity = broker.get_account_equity()
        logger.info(f"Account equity: ${equity:,.2f}")
        
        # Test data access
        df = broker.get_bars(SYMBOL, limit=50)
        logger.info(f"Successfully retrieved {len(df)} bars for {SYMBOL}")
        
        # Test indicator calculations
        test_rsi = rsi(df['close']).iloc[-1]
        test_sma = sma(df['close'], 20).iloc[-1]
        logger.info(f"Sample indicators - RSI: {test_rsi:.1f}, SMA20: {test_sma:.2f}")
        
        logger.info("Health check passed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def display_startup_info():
    """Display configuration and startup information"""
    logger.info("=" * 60)
    logger.info("Enhanced AMD Swing Trader v2.0")
    logger.info("=" * 60)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'}")
    logger.info(f"Risk per trade: {RISK_PER_TRADE:.1%}")
    logger.info(f"Max position risk: {MAX_POSITION_RISK_PCT:.1%}")
    logger.info(f"Daily loss limit: {DAILY_LOSS_LIMIT:.1%}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    logger.info(f"Take profit multiplier: {TAKE_PROFIT_MULTIPLIER}x")
    logger.info(f"Alpaca base URL: {ALPACA_BASE}")
    
    if tradeapi:
        logger.info("✓ Alpaca Trade API available")
    else:
        logger.warning("⚠ Alpaca Trade API not installed (dry-run only)")
    
    if ALPACA_KEY and ALPACA_SECRET:
        logger.info("✓ Alpaca credentials configured")
    else:
        logger.info("ℹ No Alpaca credentials (simulation mode)")
    
    logger.info("=" * 60)

def main():
    """Main entry point with comprehensive setup"""
    try:
        display_startup_info()
        
        # Perform health check
        if not run_health_check():
            logger.error("Health check failed, exiting")
            sys.exit(1)
        
        # Validate configuration
        config = TradingConfig()
        config.validate()
        
        if not DRY_RUN:
            confirmation = input("You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
            if confirmation != 'CONFIRM':
                logger.info("Live trading cancelled by user")
                sys.exit(0)
        
        logger.info("Starting trading bot...")
        main_loop()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    main()