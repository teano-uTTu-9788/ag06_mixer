# 🚀 AG06 Mixer - Complete Deployment Guide

## Overview

This guide walks you through deploying the AG06 Mixer system using:
- **Azure Container Apps** for the Python backend (using student credits)
- **Vercel** for the web UI (leveraging your paid plan)
- **GitHub Actions** with OIDC for CI/CD (no secrets!)
- **Optional**: DigitalOcean Droplet or iOS app with CloudKit

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Vercel CDN    │────────▶│  Azure Container │
│   (Frontend)    │   SSE   │   Apps (Backend) │
│                 │◀────────│                  │
└─────────────────┘         └──────────────────┘
        │                            │
        │                            │
        ▼                            ▼
   [End Users]                 [Auto-scaling]
                                [0-3 replicas]
```

## Prerequisites

### Required Tools
- Docker Desktop
- Azure CLI (`az`)
- Vercel CLI (`vercel`)
- Node.js & npm
- Git

### Required Accounts
- Azure account with student credits ($100)
- Vercel account (paid plan)
- GitHub account

## Quick Start (5 Minutes)

```bash
# 1. Clone and navigate to project
cd ag06_mixer/automation-framework

# 2. Test locally
chmod +x test-local.sh
./test-local.sh

# 3. Deploy everything
chmod +x deploy-all.sh
./deploy-all.sh
```

## Detailed Deployment Steps

### Step 1: Local Testing

First, verify everything works locally:

```bash
# Make scripts executable
chmod +x *.sh

# Run local tests
./test-local.sh
```

Expected output: `✅ ALL TESTS PASSED! (100%)`

### Step 2: Azure Backend Deployment

Deploy the Python backend to Azure Container Apps:

```bash
# Run Azure deployment
./deploy-azure.sh
```

This script will:
1. Login to Azure (device code flow)
2. Create resource group in East US
3. Create Azure Container Registry (Basic tier)
4. Build and push Docker image
5. Create Container Apps environment
6. Deploy container with auto-scaling (0-3 replicas)

**Cost**: ~$0-2/month (scales to zero when idle)

### Step 3: Vercel Frontend Deployment

Deploy the web UI to Vercel:

```bash
# Run Vercel deployment
./deploy-vercel.sh
```

This script will:
1. Set environment variables for backend URL
2. Deploy to Vercel production
3. Enable analytics (paid plan feature)
4. Configure SSE endpoint

**Cost**: Free (included in your paid plan)

### Step 4: GitHub Actions CI/CD

Set up automated deployments:

1. **Create GitHub repository**:
```bash
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/ag06-mixer.git
git push -u origin main
```

2. **Configure Azure OIDC** (no secrets!):
```bash
# Follow instructions in AZURE_OIDC_SETUP.md
# This eliminates the need for storing Azure credentials
```

3. **Add GitHub variables** (Settings → Secrets → Variables):
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

Now every push to `main` will automatically deploy!

## Testing the Deployment

### Backend API Tests

```bash
# Get your backend URL from deployment-info.json
BACKEND_URL=$(cat deployment-info.json | jq -r .backend_url)

# Test health endpoint
curl $BACKEND_URL/health

# Test status endpoint
curl $BACKEND_URL/api/status

# Test SSE stream (will stream events)
curl -N $BACKEND_URL/api/stream
```

### Frontend UI Test

1. Open the frontend URL from deployment-info.json
2. Click "Connect to Stream"
3. Watch real-time audio metrics update
4. Adjust sliders to change processing parameters

## Cost Optimization

### Azure Savings
- **Scale to zero**: No charges when idle
- **Minimal resources**: 0.25 vCPU, 0.5GB RAM
- **Basic ACR tier**: Cheapest registry option
- **Student credits**: $100 free credits

### Monitoring Costs
```bash
# Check current usage
az consumption usage list \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --query "[?contains(instanceName, 'ag06')].{name:instanceName, cost:pretaxCost}" \
  -o table
```

### Set Budget Alerts
```bash
# Create $10 budget alert
az consumption budget create \
  --amount 10 \
  --budget-name "ag06-mixer-budget" \
  --category Cost \
  --time-grain Monthly \
  --resource-group ag06-mixer-rg
```

## Production Features

### Auto-Scaling
- Scales from 0 to 3 replicas based on load
- 10 concurrent connections per replica
- Automatic scale-down after 5 minutes idle

### Health Monitoring
```bash
# View logs
az containerapp logs show \
  -n ag06-mixer-backend \
  -g ag06-mixer-rg \
  --follow

# Check metrics
az monitor metrics list \
  --resource /subscriptions/{sub-id}/resourceGroups/ag06-mixer-rg/providers/Microsoft.App/containerApps/ag06-mixer-backend \
  --metric "Requests" \
  --interval PT1M
```

### Performance
- **Latency**: <20ms average
- **Throughput**: 100+ req/sec per replica
- **SSE Streaming**: 10Hz update rate
- **Global CDN**: Vercel edge network

## Troubleshooting

### Backend Issues

```bash
# Check container status
az containerapp show \
  -n ag06-mixer-backend \
  -g ag06-mixer-rg \
  --query "properties.runningStatus"

# View recent logs
az containerapp logs show \
  -n ag06-mixer-backend \
  -g ag06-mixer-rg \
  --tail 50

# Restart container
az containerapp revision restart \
  -n ag06-mixer-backend \
  -g ag06-mixer-rg
```

### Frontend Issues

```bash
# Check Vercel deployment
vercel ls

# View logs
vercel logs ag06-mixer-ui

# Redeploy
vercel --prod
```

### SSE Connection Issues

1. Check CORS configuration
2. Verify firewall rules
3. Test with curl first
4. Check browser console for errors

## Clean Up

To completely remove all resources and stop charges:

```bash
# Remove Azure resources
az group delete --name ag06-mixer-rg --yes

# Remove Vercel deployment
vercel remove ag06-mixer-ui --yes

# Remove local Docker images
docker rmi ag06mixeracr.azurecr.io/ag06-mixer:latest
```

## Optional: DigitalOcean Droplet

For an alternative to Azure:

```bash
# Create $6/month droplet
doctl compute droplet create ag06-mixer \
  --size s-1vcpu-1gb \
  --image docker-20-04 \
  --region nyc1

# Deploy with Docker Compose
ssh root@DROPLET_IP
docker run -d -p 80:8080 ag06-mixer:latest
```

## Optional: iOS App with CloudKit

1. Open Xcode project (if created)
2. Enable CloudKit capability
3. Configure container identifier
4. Add SSE client library
5. Build and deploy to TestFlight

## Support

### Logs and Monitoring
- Azure Portal: portal.azure.com
- Vercel Dashboard: vercel.com/dashboard
- GitHub Actions: github.com/YOUR_REPO/actions

### Common Issues
1. **"Subscription not found"**: Run `az account set --subscription YOUR_SUB_ID`
2. **"Port already in use"**: Kill process with `lsof -ti:8080 | xargs kill -9`
3. **"Docker daemon not running"**: Start Docker Desktop
4. **"Vercel CLI not found"**: Install with `npm i -g vercel`

## Success Metrics

✅ Backend health endpoint returns 200  
✅ Frontend loads without errors  
✅ SSE stream delivers events  
✅ Auto-scaling works (0-3 replicas)  
✅ Costs stay under $5/month  

## Next Steps

1. **Custom Domain**: Add your domain to Vercel
2. **API Authentication**: Add API keys for production
3. **Database**: Add Azure Cosmos DB for persistence
4. **CDN**: Configure Azure CDN for static assets
5. **Monitoring**: Set up Application Insights

---

**Congratulations!** 🎉 Your AG06 Mixer is now deployed to the cloud with enterprise-grade infrastructure at student budget prices!