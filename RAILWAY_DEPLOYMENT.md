# Railway Deployment Guide

This guide will help you deploy your FastAPI application to Railway.

## ✅ Files Created

The following files have been created for Railway deployment:

1. **Procfile** - Defines the start command for Railway
2. **railway.json** - Railway-specific configuration
3. **nixpacks.toml** - Build configuration for better dependency management
4. **.railwayignore** - Excludes unnecessary files from deployment
5. **main.py** - Updated to use `$PORT` environment variable and Railway domain in CORS

## 🚀 Deployment Steps

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### Step 2: Sign Up for Railway

1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign up with your GitHub account
4. Authorize Railway to access your repositories

### Step 3: Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository: `thisisshivamtiwari/excelllm-be`
4. Railway will automatically detect it's a Python project

### Step 4: Configure Environment Variables

In the Railway dashboard, go to your project → **Variables** tab and add:

```
MONGODB_URI=your-mongodb-atlas-connection-string
JWT_SECRET_KEY=your-secret-key-minimum-32-characters-long
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
GROQ_API_KEY=your-groq-api-key
GOOGLE_API_KEY=your-google-gemini-api-key
CORS_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com
ENVIRONMENT=production
PORT=8080
DEBUG=false
```

**Important Notes:**
- Replace all placeholder values with your actual credentials
- `MONGODB_URI` should be your MongoDB Atlas connection string
- `JWT_SECRET_KEY` must be at least 32 characters
- `CORS_ORIGINS` should include your frontend URL(s)

### Step 5: Configure Service Settings

1. Go to **Settings** tab in your Railway project
2. **Service Name**: `excelllm-be` (or your preferred name)
3. **Region**: Choose the region closest to your users
4. Railway will auto-detect:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 6: Set Up Custom Domain (Optional)

1. Go to **Settings** → **Networking**
2. Click "Generate Domain" to get a Railway domain
   - Example: `excelllm-be-production.up.railway.app`
3. Or add your own custom domain
4. Railway automatically sets `RAILWAY_PUBLIC_DOMAIN` environment variable
5. The app will automatically add this domain to CORS allowed origins

### Step 7: Deploy

Railway will automatically:
1. Build your application
2. Install dependencies from `requirements.txt`
3. Start the application using the Procfile
4. Deploy to the cloud

Watch the deployment logs in the Railway dashboard to see the build progress.

### Step 8: Test Your Deployment

Once deployed, visit:
- **API Docs**: `https://your-app-name.up.railway.app/docs`
- **Health Check**: `https://your-app-name.up.railway.app/` (if you have a root endpoint)

## 📋 MongoDB Atlas Setup

Since Railway doesn't provide MongoDB, you need MongoDB Atlas:

1. **Create MongoDB Atlas Account**
   - Go to https://www.mongodb.com/cloud/atlas
   - Sign up for free (M0 cluster)

2. **Create Cluster**
   - Choose free tier (M0)
   - Select region closest to Railway region
   - Create cluster (takes 3-5 minutes)

3. **Create Database User**
   - Go to Database Access
   - Add New Database User
   - Username/Password authentication
   - Save credentials securely

4. **Whitelist IP Addresses**
   - Go to Network Access
   - Add IP Address
   - For Railway, use `0.0.0.0/0` (allows all IPs) or Railway's IP ranges
   - Click "Allow Access from Anywhere" for development

5. **Get Connection String**
   - Go to Database → Connect
   - Choose "Connect your application"
   - Copy connection string
   - Replace `<password>` with your database user password
   - Replace `<dbname>` with your database name (optional)
   - Example: `mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority`

6. **Add to Railway**
   - Paste connection string into `MONGODB_URI` environment variable

## 🔍 Monitoring & Logs

### View Logs
- Railway Dashboard → Your Project → **Deployments** → Click on deployment → **View Logs**
- Or use Railway CLI: `railway logs`

### Metrics
- Railway Dashboard shows:
  - CPU usage
  - Memory usage
  - Network traffic
  - Request count

### Alerts
- Set up notifications in **Settings** → **Notifications**
- Get alerts for deployment failures, high resource usage, etc.

## 🛠️ Troubleshooting

### Build Fails

**Issue**: Build fails during dependency installation
- **Solution**: Check build logs for specific error
- Common issues:
  - Large dependencies (sentence-transformers) may timeout
  - Python version mismatch
  - Missing system dependencies

**Fix**: 
```bash
# Check requirements.txt has all dependencies
# Consider pinning versions for stability
```

### App Crashes on Startup

**Issue**: App starts but immediately crashes
- **Solution**: Check runtime logs
- Common causes:
  - Missing environment variables
  - MongoDB connection issues
  - Port binding problems

**Fix**:
```bash
# Verify all environment variables are set
# Check MongoDB connection string is correct
# Ensure MongoDB IP whitelist includes Railway IPs
```

### Import Errors

**Issue**: ModuleNotFoundError
- **Solution**: Ensure all dependencies are in `requirements.txt`
- Check Python path in logs

### Slow Cold Starts

**Issue**: First request after inactivity is slow
- **Solution**: This is normal on free tier
- Consider upgrading to Hobby plan ($5/month) for better performance

### CORS Errors

**Issue**: Frontend can't connect to API
- **Solution**: 
  1. Check `CORS_ORIGINS` includes your frontend URL
  2. Verify Railway domain is in allowed origins (auto-added)
  3. Check browser console for specific CORS error

## 💰 Railway Pricing

- **Free Tier**: $5 credit/month (usually enough for small apps)
- **Hobby Plan**: $5/month (better performance, no cold starts)
- **Pro Plan**: $20/month (production features, better support)

## 📝 Quick Reference

### Railway CLI (Optional)

Install Railway CLI for easier management:
```bash
npm i -g @railway/cli
railway login
railway link  # Link to your project
railway up    # Deploy
railway logs  # View logs
```

### Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB Atlas connection string | Yes |
| `JWT_SECRET_KEY` | Secret key for JWT tokens (min 32 chars) | Yes |
| `JWT_ALGORITHM` | JWT algorithm (default: HS256) | No |
| `GROQ_API_KEY` | Groq API key for LLM | Yes |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `CORS_ORIGINS` | Comma-separated frontend URLs | Yes |
| `ENVIRONMENT` | Set to "production" | Yes |
| `PORT` | Server port (Railway sets automatically) | Auto |
| `RAILWAY_PUBLIC_DOMAIN` | Railway domain (auto-set) | Auto |

### Useful URLs

- **Railway Dashboard**: https://railway.app/dashboard
- **Your API**: `https://your-app-name.up.railway.app`
- **API Docs**: `https://your-app-name.up.railway.app/docs`
- **MongoDB Atlas**: https://cloud.mongodb.com

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Railway account created
- [ ] Project created and connected to GitHub
- [ ] All environment variables set
- [ ] MongoDB Atlas cluster created
- [ ] MongoDB connection string added to Railway
- [ ] MongoDB IP whitelist configured
- [ ] First deployment successful
- [ ] API tested at `/docs` endpoint
- [ ] CORS configured correctly
- [ ] Frontend can connect to API
- [ ] Logs checked for errors

## 🎉 Next Steps

1. **Test all endpoints** using the `/docs` interface
2. **Connect your frontend** to the Railway API URL
3. **Monitor usage** to stay within free tier limits
4. **Set up custom domain** if needed
5. **Configure auto-deploy** from GitHub (already enabled by default)

## 📞 Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Railway Status: https://status.railway.app

---

**Happy Deploying! 🚀**

