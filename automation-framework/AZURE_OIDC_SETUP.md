# Azure OIDC Setup for GitHub Actions

This guide helps you set up OpenID Connect (OIDC) authentication between GitHub Actions and Azure, eliminating the need for storing client secrets.

## Prerequisites

- Azure account with student credits ($100)
- GitHub repository for AG06 Mixer
- Azure CLI installed locally

## Step 1: Create Azure Service Principal

```bash
# Login to Azure
az login --use-device-code

# Create a service principal and configure OIDC
az ad sp create-for-rbac \
  --name "github-ag06mixer-oidc" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/ag06-mixer-rg \
  --sdk-auth
```

Save the output - you'll need:
- `clientId`
- `tenantId`
- `subscriptionId`

## Step 2: Configure Federated Credentials

```bash
# Get the Object ID of the service principal
OBJECT_ID=$(az ad sp list --display-name "github-ag06mixer-oidc" --query "[0].id" -o tsv)

# Create federated credential for main branch
az ad app federated-credential create \
  --id $OBJECT_ID \
  --parameters '{
    "name": "github-ag06mixer-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:YOUR_GITHUB_USERNAME/ag06-mixer:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'

# Create federated credential for pull requests (optional)
az ad app federated-credential create \
  --id $OBJECT_ID \
  --parameters '{
    "name": "github-ag06mixer-pr",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:YOUR_GITHUB_USERNAME/ag06-mixer:pull_request",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

## Step 3: Configure GitHub Repository

### Add Repository Variables (not secrets!)

Go to your GitHub repository:
1. Settings → Secrets and variables → Actions
2. Click on "Variables" tab
3. Add these **repository variables**:

- `AZURE_CLIENT_ID`: Your service principal's clientId
- `AZURE_TENANT_ID`: Your Azure tenant ID
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

**Note**: These are not sensitive and can be stored as variables, not secrets!

## Step 4: Test the Setup

1. Push to main branch or create a PR
2. Check GitHub Actions workflow
3. Verify deployment succeeds without any stored secrets

## Benefits of OIDC

✅ **No secrets to rotate** - Uses short-lived tokens  
✅ **More secure** - No long-lived credentials stored  
✅ **Easier compliance** - Better audit trail  
✅ **Automatic** - GitHub handles token exchange  

## Troubleshooting

### Error: "No subscription found"
- Ensure the service principal has Contributor role on the resource group
- Check that subscription ID is correct

### Error: "AADSTS70021: No matching federated identity record"
- Verify the repository name in federated credential matches exactly
- Check that branch name (ref:refs/heads/main) is correct
- Ensure you're using the correct GitHub username

### Error: "Insufficient privileges"
```bash
# Grant proper permissions
az role assignment create \
  --assignee {service-principal-id} \
  --role "Contributor" \
  --scope /subscriptions/{subscription-id}/resourceGroups/ag06-mixer-rg
```

## Cost Optimization

To minimize Azure costs with student credits:

1. **Scale to Zero**: Container Apps configured to scale to 0 replicas when idle
2. **Minimal Resources**: Using 0.25 vCPU and 0.5GB RAM
3. **Basic Tier**: ACR using Basic tier (cheapest option)
4. **Auto-shutdown**: Consider adding time-based scaling rules

## Clean Up Resources

To stop all charges:

```bash
# Delete the entire resource group
az group delete --name ag06-mixer-rg --yes

# Remove federated credentials
az ad app federated-credential delete \
  --id $OBJECT_ID \
  --federated-credential-id "github-ag06mixer-main"
```

## Next Steps

1. ✅ Push code to trigger deployment
2. ✅ Monitor costs in Azure Portal
3. ✅ Set up budget alerts ($10 warning, $20 critical)
4. ✅ Configure custom domain (optional)