# Enhanced Trading Bot v4.0

A clean, optimized multi-stock swing trading bot with Discord notifications, PostgreSQL logging, and web dashboard. Designed for both local development and Railway cloud deployment.

## ✨ Features

- **Multi-stock swing trading** with technical analysis
- **Discord webhook notifications** for all trading activities  
- **PostgreSQL database logging** for activity tracking
- **Web dashboard** for real-time monitoring
- **Railway cloud deployment** ready
- **Clean, optimized codebase** with proper error handling

## 🚀 Quick Start

### Local Development

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd trading-bot
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Setup database (optional)**
   ```bash
   python run.py setup
   ```

4. **Start the bot**
   ```bash
   python run.py bot          # Start trading bot
   python run.py dashboard    # Start web dashboard
   ```

### Railway Deployment

1. **Fork this repository**

2. **Deploy to Railway**
   - Connect your GitHub repo to Railway
   - Add environment variables in Railway dashboard
   - Deploy automatically

3. **Required Railway Environment Variables**
   ```
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_API_SECRET=your_alpaca_secret
   DISCORD_WEBHOOK_URL=your_discord_webhook
   DRY_RUN=true  # Set to false for live trading
   ```

## 📋 Environment Variables

### Required
- `ALPACA_API_KEY` - Alpaca Markets API key
- `ALPACA_API_SECRET` - Alpaca Markets secret key

### Optional
- `ALPACA_BASE_URL` - API endpoint (default: paper trading)
- `DISCORD_WEBHOOK_URL` - Discord webhook for notifications
- `DATABASE_URL` - PostgreSQL connection string
- `WATCHLIST` - Comma-separated stock symbols (default: AMD,NVDA,AAPL,MSFT,GOOGL)
- `DRY_RUN` - Enable dry-run mode (default: true)
- `CHECK_INTERVAL` - Scan interval in seconds (default: 300)
- `RISK_PER_TRADE` - Risk percentage per trade (default: 0.01)

## 🏗️ Architecture

```
trading_bot.py          # Main trading bot implementation
dashboard.py            # Web dashboard with Flask
setup_database.py       # Database setup script
run.py                  # Clean launcher script
requirements.txt        # Optimized dependencies
templates/              # Dashboard HTML templates
```

## 📊 Trading Strategy

The bot implements a **fibonacci retracement swing trading strategy**:

1. **Trend Analysis**: Uses 50/200 SMA for trend direction
2. **Entry Signals**: Fibonacci retracement levels (50%, 61.8%) + RSI confirmation
3. **Risk Management**: Position sizing based on stop-loss distance
4. **Exit Strategy**: Take profit at 2:1 risk/reward ratio

## 🔒 Security & Risk Management

- **Paper trading by default** - Set `DRY_RUN=false` for live trading
- **Position sizing** based on account equity and risk percentage
- **Stop-loss orders** for all positions
- **Daily loss limits** to prevent significant drawdowns
- **Error handling** with automatic recovery

## 🛠️ Development

### Project Structure
```
├── trading_bot.py              # Core trading logic
├── dashboard.py                # Web interface
├── setup_database.py           # Database setup
├── run.py                      # Application launcher
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
├── templates/
│   └── dashboard.html         # Dashboard template
├── Dockerfile                 # Docker configuration
├── railway.json              # Railway deployment config
└── README.md                 # This file
```

### Code Quality Standards

- **Clean imports** with graceful fallbacks
- **Type hints** for better code clarity
- **Error handling** for all external dependencies
- **Logging** for debugging and monitoring
- **Memory optimization** for cloud deployment
- **Modular design** for easy maintenance

### Testing

```bash
# Run system health check
python run.py test

# Check database connectivity
python setup_database.py

# Start dashboard for testing
python run.py dashboard
```

## 📈 Dashboard Features

Access the web dashboard at `http://localhost:5000`:

- **Portfolio Overview** - Current positions and P&L
- **Performance Charts** - Historical portfolio performance
- **Trading Signals** - Recent buy/sell signals
- **Bot Status** - Uptime and error monitoring
- **Market Data** - Real-time price information

## 🚨 Notifications

Discord notifications include:

- **🚀 Bot Started** - Startup confirmation
- **💰 Purchase Executed** - Trade confirmations
- **📊 Daily Updates** - Portfolio performance
- **🚨 Error Alerts** - System issues
- **🛑 Bot Stopped** - Shutdown notifications

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real money.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check the documentation
- Review the logs for error details

---

**Enhanced Trading Bot v4.0** - Clean, optimized, and production-ready.
