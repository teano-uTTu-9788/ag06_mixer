# AG06 Mixer Enterprise Production Deployment

This package contains everything needed to deploy AG06 Mixer to production on AWS.

## Quick Start

1. **Prerequisites**
   ```bash
   # Install required tools
   - AWS CLI v2
   - Terraform >= 1.0
   - kubectl
   - Helm v3
   - Docker
   ```

2. **Configure AWS Credentials**
   ```bash
   aws configure
   ```

3. **Deploy Infrastructure**
   ```bash
   ./deployment-scripts/setup-infrastructure.sh
   ./deployment-scripts/deploy.sh
   ```

## Architecture Overview

### Cloud Infrastructure
- **VPC**: Custom VPC with public/private subnets across 3 AZs
- **EKS**: Managed Kubernetes cluster with auto-scaling node groups
- **RDS**: PostgreSQL with Multi-AZ deployment and read replicas
- **ALB**: Application Load Balancer with SSL termination
- **Route53**: DNS management and SSL certificate validation

### Application Stack
- **Kubernetes**: Container orchestration with auto-scaling
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting dashboards
- **AWS Load Balancer Controller**: Kubernetes ingress management

## Directory Structure

```
├── infrastructure/           # Terraform infrastructure code
├── k8s-manifests/           # Kubernetes application manifests
├── deployment-scripts/      # Deployment automation scripts
└── deployment-docs/         # Documentation and guides
```

## Deployment Steps

### 1. Infrastructure Setup
```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```

### 2. Application Deployment
```bash
kubectl apply -f k8s-manifests/
```

### 3. Monitoring Setup
```bash
./deployment-scripts/setup-monitoring.sh
```

## Configuration

### Domain Configuration
- Primary Domain: ag06mixer.com
- API Endpoint: api.ag06mixer.com
- Monitoring: monitor.ag06mixer.com

### Resource Configuration
- Region: us-east-1
- Environment: production
- App Replicas: 3
- Database: db.r5.large
- Compute: t3.medium

## Monitoring and Observability

### Grafana Dashboard
```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```
Access at: http://localhost:3000 (admin/admin123)

### Prometheus Metrics
```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### Application Logs
```bash
kubectl logs -f deployment/ag06mixer-app -n ag06mixer
```

## Security

### SSL/TLS
- Automatic SSL certificate provisioning via AWS ACM
- HTTP to HTTPS redirection
- TLS 1.2+ enforcement

### Network Security
- VPC isolation with public/private subnets
- Security groups with least privilege access
- Network policies for pod-to-pod communication
- WAF integration (optional)

### Secrets Management
- AWS Secrets Manager for database credentials
- Kubernetes secrets for application configuration
- IAM roles for service accounts (IRSA)

## High Availability

### Multi-AZ Deployment
- Applications across multiple availability zones
- Database with Multi-AZ failover
- Load balancing across healthy instances

### Auto-Scaling
- Horizontal Pod Autoscaler (CPU/Memory)
- Cluster Autoscaler for nodes
- Application Load Balancer for traffic distribution

### Backup and Recovery
- Automated RDS backups (30-day retention)
- Point-in-time recovery capability
- Cross-region backup replication

## Cost Optimization

### Resource Right-sizing
- T3 instances for cost-effective compute
- GP3 storage for better price/performance
- Reserved instances for predictable workloads

### Auto-Scaling
- Scale down during low usage periods
- Spot instances for non-critical workloads
- Resource requests/limits for efficient scheduling

## Troubleshooting

### Common Issues

1. **DNS Propagation Delays**
   ```bash
   # Check DNS resolution
   dig ag06mixer.com
   ```

2. **SSL Certificate Validation**
   ```bash
   # Check certificate status
   aws acm describe-certificate --certificate-arn <ARN>
   ```

3. **Pod Startup Issues**
   ```bash
   # Check pod status
   kubectl describe pod <pod-name> -n ag06mixer
   ```

### Log Aggregation
```bash
# Application logs
kubectl logs -f deployment/ag06mixer-app -n ag06mixer

# System logs
kubectl logs -f -l app=aws-load-balancer-controller -n kube-system
```

## Maintenance

### Updates and Patches
- Kubernetes version updates
- Application deployment updates
- Security patches and vulnerability fixes

### Monitoring Alerts
- Application health checks
- Database connection monitoring
- Resource utilization alerts
- SSL certificate expiration warnings

## Support

For issues and questions:
1. Check troubleshooting guide above
2. Review application logs
3. Check AWS CloudWatch for infrastructure metrics
4. Contact development team for application-specific issues

## License

This deployment package is part of the AG06 Mixer Enterprise system.
