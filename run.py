#!/usr/bin/env python3
"""
Enhanced Trading Bot Launcher

Clean launcher for the enhanced trading bot system.

Usage:
    python run.py [component]

Components:
    bot        - Start the main trading bot (default)
    dashboard  - Start the web dashboard
    setup      - Setup database tables

Examples:
    python run.py              # Start trading bot
    python run.py dashboard    # Start dashboard only
    python run.py setup        # Setup database
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_bot():
    """Start the main trading bot"""
    print("Starting Enhanced Trading Bot v4.0...")
    
    try:
        import trading_bot
        trading_bot.main()
    except ImportError as e:
        print(f"Error importing bot module: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting bot: {e}")
        sys.exit(1)

def run_dashboard():
    """Start the web dashboard"""
    print("Starting Trading Dashboard...")
    
    try:
        import dashboard
        port = int(os.getenv("PORT", "5000"))
        dashboard.app.run(host="0.0.0.0", port=port, debug=False)
    except ImportError as e:
        print(f"Error importing dashboard module: {e}")
        print("Make sure Flask is installed: pip install flask flask-cors")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

def setup_database():
    """Setup database tables"""
    print("Setting up database...")
    
    try:
        import setup_database
        setup_database.main()
    except ImportError as e:
        print(f"Error importing setup module: {e}")
        print("Make sure psycopg2 is installed: pip install psycopg2-binary")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    component = sys.argv[1] if len(sys.argv) > 1 else "bot"
    
    if component == "bot":
        run_bot()
    elif component == "dashboard":
        run_dashboard()
    elif component == "setup":
        setup_database()
    else:
        print(f"Unknown component: {component}")
        print("Available components: bot, dashboard, setup")
        sys.exit(1)

if __name__ == "__main__":
    main()
