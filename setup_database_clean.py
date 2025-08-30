#!/usr/bin/env python3
"""
Database Setup Script for Enhanced Trading Bot

Clean, optimized database setup for both local and Railway deployment.

Author: Enhanced Trading Bot System
Version: 4.0.0
"""

import os
import sys

# Import handling for optional dependencies
try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    POSTGRES_AVAILABLE = False

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None

# Local database config (if not using DATABASE_URL)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_bot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

def create_connection():
    """Create database connection"""
    if not POSTGRES_AVAILABLE:
        print("Error: psycopg2 not available. Install with: pip install psycopg2-binary")
        return None
    
    try:
        if DATABASE_URL:
            # Railway or other cloud provider
            conn = psycopg2.connect(DATABASE_URL)
            print("Connected using DATABASE_URL")
        else:
            # Local development
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            print(f"Connected to local database: {POSTGRES_DB}")
        
        return conn
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def create_tables(conn):
    """Create required database tables"""
    tables = {
        'trading_activity': """
            CREATE TABLE IF NOT EXISTS trading_activity (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(10) NOT NULL,
                action VARCHAR(50) NOT NULL,
                quantity INTEGER DEFAULT 0,
                price DECIMAL(10,2) DEFAULT 0,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        
        'portfolio_snapshots': """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_value DECIMAL(12,2) NOT NULL,
                cash_value DECIMAL(12,2) NOT NULL,
                positions_value DECIMAL(12,2) NOT NULL,
                daily_pnl DECIMAL(10,2) DEFAULT 0,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        
        'bot_status': """
            CREATE TABLE IF NOT EXISTS bot_status (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) NOT NULL,
                message TEXT,
                uptime_seconds INTEGER DEFAULT 0,
                errors_count INTEGER DEFAULT 0,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    }
    
    cursor = conn.cursor()
    
    for table_name, table_sql in tables.items():
        try:
            cursor.execute(table_sql)
            print(f"✓ Created/verified table: {table_name}")
        except Exception as e:
            print(f"✗ Error creating table {table_name}: {e}")
            return False
    
    # Create indexes for better performance
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trading_activity_symbol ON trading_activity(symbol);",
        "CREATE INDEX IF NOT EXISTS idx_trading_activity_timestamp ON trading_activity(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_bot_status_timestamp ON bot_status(timestamp);"
    ]
    
    for index_sql in indexes:
        try:
            cursor.execute(index_sql)
        except Exception as e:
            print(f"Warning: Index creation failed: {e}")
    
    conn.commit()
    cursor.close()
    print("✓ All tables and indexes created successfully")
    return True

def test_database_operations(conn):
    """Test basic database operations"""
    try:
        cursor = conn.cursor()
        
        # Test insert
        cursor.execute("""
            INSERT INTO trading_activity (symbol, action, metadata)
            VALUES (%s, %s, %s)
        """, ('TEST', 'DATABASE_SETUP', '{"test": true}'))
        
        # Test select
        cursor.execute("""
            SELECT COUNT(*) FROM trading_activity 
            WHERE action = 'DATABASE_SETUP'
        """)
        count = cursor.fetchone()[0]
        
        # Clean up test data
        cursor.execute("DELETE FROM trading_activity WHERE action = 'DATABASE_SETUP'")
        
        conn.commit()
        cursor.close()
        
        print("✓ Database operations test passed")
        return True
        
    except Exception as e:
        print(f"✗ Database operations test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Enhanced Trading Bot - Database Setup v4.0")
    print("=" * 50)
    
    if not POSTGRES_AVAILABLE:
        print("PostgreSQL support not available.")
        print("Install with: pip install psycopg2-binary")
        sys.exit(1)
    
    if IS_RAILWAY:
        print("Railway environment detected")
    else:
        print("Local environment detected")
    
    # Create connection
    conn = create_connection()
    if not conn:
        print("Database setup failed - could not connect")
        sys.exit(1)
    
    try:
        # Create tables
        if not create_tables(conn):
            print("Database setup failed - could not create tables")
            sys.exit(1)
        
        # Test operations
        if not test_database_operations(conn):
            print("Database setup completed with warnings")
        else:
            print("✓ Database setup completed successfully!")
        
        conn.close()
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
