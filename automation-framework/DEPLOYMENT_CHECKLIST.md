# AG06 Mixer Production Deployment Checklist

## Pre-Deployment Checklist

### AWS Account Setup
- [ ] AWS CLI installed and configured
- [ ] Appropriate IAM permissions for deployment
- [ ] Domain registered or available (ag06mixer.com)
- [ ] Route53 hosted zone ready

### Tools Installation
- [ ] Terraform >= 1.0 installed
- [ ] kubectl installed
- [ ] Helm v3 installed
- [ ] Docker installed (for building images)

### Configuration
- [ ] Update S3 bucket name in infrastructure/main.tf
- [ ] Review variables.tf for environment-specific settings
- [ ] Update secrets in k8s-manifests/03-secrets.yaml
- [ ] Configure monitoring alert recipients

## Deployment Steps

### Phase 1: Infrastructure
- [ ] Run `./deployment-scripts/setup-infrastructure.sh`
- [ ] Update Terraform backend configuration
- [ ] Execute `terraform plan` and review changes
- [ ] Execute `terraform apply` to create infrastructure
- [ ] Verify infrastructure outputs

### Phase 2: Kubernetes Setup
- [ ] Configure kubectl for EKS cluster
- [ ] Install AWS Load Balancer Controller
- [ ] Apply Kubernetes manifests
- [ ] Verify pod deployment and health

### Phase 3: Application Deployment
- [ ] Build and push application Docker images
- [ ] Update image references in deployment manifests
- [ ] Deploy application to Kubernetes
- [ ] Configure ingress and SSL certificates

### Phase 4: Monitoring
- [ ] Install Prometheus and Grafana
- [ ] Configure monitoring dashboards
- [ ] Set up alerting rules
- [ ] Test monitoring endpoints

## Post-Deployment Verification

### Application Health
- [ ] Verify application health endpoint: https://ag06mixer.com/health
- [ ] Test API endpoints: https://api.ag06mixer.com
- [ ] Check SSL certificate validity
- [ ] Verify DNS resolution

### Infrastructure Health
- [ ] Check EKS cluster status
- [ ] Verify database connectivity
- [ ] Test load balancer functionality
- [ ] Confirm auto-scaling behavior

### Security Verification
- [ ] Verify network policies are active
- [ ] Check security groups and NACLs
- [ ] Confirm secrets are properly encrypted
- [ ] Test RBAC permissions

### Monitoring and Observability
- [ ] Access Grafana dashboards
- [ ] Verify metrics collection
- [ ] Test alerting system
- [ ] Check log aggregation

## Production Readiness

### Performance Testing
- [ ] Load testing with expected traffic
- [ ] Database performance validation
- [ ] CDN and caching verification
- [ ] Auto-scaling threshold testing

### Disaster Recovery
- [ ] Backup verification
- [ ] Recovery procedure testing
- [ ] Multi-AZ failover testing
- [ ] Documentation of recovery procedures

### Operations
- [ ] Monitoring runbooks created
- [ ] Incident response procedures
- [ ] On-call rotation setup
- [ ] Maintenance windows scheduled

## Go-Live Checklist

### Final Verification
- [ ] All health checks passing
- [ ] Performance requirements met
- [ ] Security scan completed
- [ ] Stakeholder sign-off obtained

### DNS Cutover
- [ ] Update DNS records to point to new infrastructure
- [ ] Monitor traffic transition
- [ ] Verify application functionality
- [ ] Confirm all integrations working

### Post Go-Live
- [ ] Monitor application performance
- [ ] Watch for any alerts or issues
- [ ] Communicate successful deployment
- [ ] Schedule post-deployment review

## Rollback Plan

### Immediate Rollback (if needed)
- [ ] Revert DNS changes
- [ ] Scale down new deployment
- [ ] Restore previous version
- [ ] Communicate rollback to stakeholders

### Infrastructure Rollback
- [ ] Use Terraform to revert infrastructure changes
- [ ] Restore database from backup if needed
- [ ] Update monitoring to reflect rollback
- [ ] Document rollback reasons and lessons learned

---

**Deployment Team**: _________________ **Date**: _________________

**Deployment Lead**: _________________ **Approved By**: _________________
