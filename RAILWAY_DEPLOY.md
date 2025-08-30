# Railway Deployment Guide

## 🚀 Quick Railway Deploy

### 1. Prepare Your Repository

```bash
# Ensure you have these files:
ls -la
# Should show:
# - railway_bot.py (main optimized bot)
# - railway_dashboard.py (lightweight dashboard)
# - requirements-railway.txt (minimal dependencies)
# - railway.json (Railway configuration)
```

### 2. Deploy to Railway

#### Option A: GitHub Integration (Recommended)
1. Push your code to GitHub
2. Go to [Railway.app](https://railway.app)
3. Click "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy

#### Option B: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

### 3. Setup PostgreSQL Database

```bash
# Add PostgreSQL service
railway add postgresql

# Railway automatically sets these environment variables:
# - DATABASE_URL
# - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
```

### 4. Configure Environment Variables

In Railway dashboard, set these variables:

#### Required Variables
```
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
DRY_RUN=true
WATCHLIST=AMD,NVDA,AAPL,MSFT
```

#### Optional Variables
```
DISCORD_WEBHOOK_URL=your_discord_webhook
RISK_PER_TRADE=0.01
CHECK_INTERVAL_SECONDS=180
```

### 5. Deploy Services

#### Deploy Trading Bot
```bash
# Set start command in Railway dashboard:
python railway_bot.py
```

#### Deploy Dashboard (Optional)
```bash
# Create new service for dashboard:
python railway_dashboard.py

# Set port to $PORT (Railway provides this)
```

## 🔧 Railway Optimizations

### Memory Usage
- **Limit**: 512MB (Railway default)
- **Optimized**: Batch processing, efficient caching
- **Monitoring**: Automatic garbage collection

### Performance Features
- **Connection Pooling**: Efficient database connections
- **Batch Processing**: Multiple stocks in small batches
- **Smart Caching**: Memory-aware data caching
- **Optimized Dependencies**: Minimal package footprint

### Cost Optimization
- **Resource Efficient**: Uses minimal CPU/memory
- **Smart Scheduling**: Reduced API calls during market hours
- **Automatic Scaling**: Scales down when not needed

## 📊 Railway Dashboard Access

After deployment, your dashboard will be available at:
```
https://your-app-name.railway.app
```

## 🛠️ Troubleshooting

### Common Issues

#### Build Fails
```bash
# Check requirements file
cat requirements-railway.txt

# Ensure all dependencies are compatible
```

#### Database Connection Issues
```bash
# Check PostgreSQL service is running
railway status

# Verify environment variables are set
railway variables
```

#### Memory Issues
```bash
# Monitor resource usage in Railway dashboard
# Reduce WATCHLIST size if needed
# Increase CHECK_INTERVAL_SECONDS
```

### Monitoring

#### Railway Logs
```bash
# View live logs
railway logs

# Follow logs
railway logs --follow
```

#### Resource Monitoring
- Check CPU/Memory usage in Railway dashboard
- Monitor database connections
- Track API call frequency

### Performance Tuning

#### For Low Memory (512MB)
```env
WATCHLIST=AMD,NVDA  # Reduce to 2-3 symbols
CHECK_INTERVAL_SECONDS=300  # Increase interval
```

#### For Better Performance (1GB+)
```env
WATCHLIST=AMD,NVDA,AAPL,MSFT,GOOGL,TSLA  # More symbols
CHECK_INTERVAL_SECONDS=120  # Faster scanning
```

## 🔐 Security Best Practices

### Environment Variables
- Never commit API keys to git
- Use Railway's built-in secrets management
- Rotate API keys periodically

### Database Security
- Railway PostgreSQL is secure by default
- Use strong passwords
- Enable connection encryption

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances for different strategies
- Use different watchlists per instance
- Load balance with Railway's built-in features

### Vertical Scaling
- Upgrade Railway plan for more resources
- Monitor usage and scale accordingly
- Use Railway's auto-scaling features

## 💰 Cost Management

### Railway Pricing
- **Hobby Plan**: $5/month (512MB RAM, 1GB storage)
- **Pro Plan**: $20/month (8GB RAM, 100GB storage)

### Cost Optimization Tips
1. Use efficient watchlists (fewer symbols)
2. Optimize check intervals
3. Use Railway's sleep feature for off-hours
4. Monitor resource usage regularly

## 🎯 Production Checklist

- [ ] Code tested locally
- [ ] Environment variables configured
- [ ] PostgreSQL service added
- [ ] Discord webhook tested
- [ ] Alpaca API keys verified
- [ ] Resource limits appropriate
- [ ] Monitoring setup
- [ ] Backup strategy in place
- [ ] Error alerting configured

## 🔄 Deployment Commands

### Initial Deploy
```bash
git add .
git commit -m "Railway deployment setup"
git push origin main
# Railway auto-deploys
```

### Update Deployment
```bash
git add .
git commit -m "Update trading bot"
git push origin main
# Railway auto-deploys changes
```

### Manual Deploy
```bash
railway up --detach
```

## 📞 Support

For Railway-specific issues:
- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)
- [Railway Status](https://status.railway.app)

For bot-specific issues:
- Check logs: `railway logs`
- Review environment variables
- Test database connectivity
- Verify API keys and permissions
