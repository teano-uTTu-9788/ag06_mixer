# Kubernetes Deployment Guide

## Application Architecture

### Microservices Design
- **Primary Application**: AG06 Mixer core services
- **API Gateway**: Request routing and authentication
- **Background Workers**: Asynchronous job processing
- **Monitoring**: Prometheus and Grafana stack

### Deployment Strategy
- **Rolling Updates**: Zero-downtime deployments
- **Blue-Green Capability**: Production deployment safety
- **Canary Releases**: Gradual feature rollout
- **Auto-Scaling**: CPU and memory-based scaling

## Kubernetes Resources

### 1. Namespace (ag06mixer)
Isolates application resources from system components

### 2. ConfigMap (ag06mixer-config)
- Environment variables
- Application configuration
- Feature flags
- Logging configuration

### 3. Secrets (ag06mixer-secrets)
- Database credentials
- JWT signing keys
- API keys and tokens
- SSL certificates

### 4. Deployment (ag06mixer-app)
- **Replicas**: 3 initial pods
- **Update Strategy**: RollingUpdate
- **Resource Limits**: 500m CPU, 1Gi memory
- **Health Checks**: Liveness and readiness probes

### 5. Service (ag06mixer-service)
- **Type**: LoadBalancer
- **Ports**: 80 (HTTP), 8081 (metrics)
- **Session Affinity**: None (stateless)

### 6. Ingress (ag06mixer-ingress)
- **Controller**: AWS Load Balancer Controller
- **SSL**: Automatic certificate management
- **Routing**: Host-based routing to services

### 7. HorizontalPodAutoscaler (ag06mixer-hpa)
- **Min Replicas**: 2
- **Max Replicas**: 10
- **Metrics**: CPU (70%), Memory (80%)
- **Scaling Policies**: Configurable scale-up/down behavior

## Security Configuration

### Service Account (ag06mixer-service-account)
- **IRSA**: IAM Roles for Service Accounts
- **Permissions**: Minimal required AWS permissions
- **Token**: Automatic token projection

### Role-Based Access Control (RBAC)
- **Role**: ag06mixer-role
- **Permissions**: Read-only access to required resources
- **Binding**: Service account to role mapping

### Network Policies
- **Ingress Rules**: Allow traffic from ingress controllers
- **Egress Rules**: Allow database and external API access
- **Isolation**: Pod-to-pod communication restrictions

### Pod Security
- **Security Context**: Non-root user execution
- **Read-Only Filesystem**: Immutable container filesystem
- **Capabilities**: Dropped ALL, minimal required capabilities
- **Privilege Escalation**: Disabled

## Application Configuration

### Environment Variables
```yaml
- ENVIRONMENT: production
- LOG_LEVEL: INFO
- METRICS_ENABLED: true
- HEALTH_CHECK_PORT: 8081
```

### Resource Management
```yaml
resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 500m
    memory: 1Gi
```

### Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Deployment Process

### 1. Prerequisites
```bash
# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name ag06mixer-cluster

# Verify connection
kubectl get nodes
```

### 2. Deploy Application
```bash
# Apply all manifests
kubectl apply -f k8s-manifests/

# Verify deployment
kubectl get pods -n ag06mixer
kubectl get services -n ag06mixer
```

### 3. Configure Ingress
```bash
# Install AWS Load Balancer Controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=ag06mixer-cluster
```

### 4. Validate Deployment
```bash
# Check application health
kubectl exec -it deployment/ag06mixer-app -n ag06mixer -- curl localhost:8080/health

# Check ingress status
kubectl get ingress -n ag06mixer
```

## Monitoring and Observability

### Prometheus Metrics
- **Application Metrics**: Custom business metrics
- **System Metrics**: CPU, memory, network, disk
- **Kubernetes Metrics**: Pod, node, and cluster metrics

### Grafana Dashboards
- **Application Dashboard**: Request rates, response times, errors
- **Infrastructure Dashboard**: Node and pod resource utilization
- **Business Dashboard**: Custom KPIs and metrics

### Log Aggregation
```bash
# Application logs
kubectl logs -f deployment/ag06mixer-app -n ag06mixer

# Previous pod logs
kubectl logs --previous deployment/ag06mixer-app -n ag06mixer
```

### Alerting Rules
- High error rates (> 5%)
- High response times (> 2s)
- Pod crash loops
- Resource exhaustion

## Scaling and Performance

### Horizontal Pod Autoscaler
```yaml
minReplicas: 2
maxReplicas: 10
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
```

### Vertical Pod Autoscaler (Optional)
- Automatic resource request/limit adjustment
- Historical usage analysis
- Rightsizing recommendations

### Cluster Autoscaler
- Automatic node scaling based on pod scheduling
- Integration with EKS managed node groups
- Cost optimization through scale-down

## Troubleshooting

### Common Issues

1. **Pod Scheduling Issues**
```bash
# Check pod events
kubectl describe pod <pod-name> -n ag06mixer

# Check node resources
kubectl top nodes
```

2. **Image Pull Issues**
```bash
# Check image pull secrets
kubectl get secrets -n ag06mixer

# Verify image registry access
kubectl describe pod <pod-name> -n ag06mixer
```

3. **Service Discovery**
```bash
# Check service endpoints
kubectl get endpoints -n ag06mixer

# Test service connectivity
kubectl run test-pod --image=busybox -it --rm -- nslookup ag06mixer-service.ag06mixer.svc.cluster.local
```

4. **Ingress Issues**
```bash
# Check ingress status
kubectl describe ingress ag06mixer-ingress -n ag06mixer

# Check AWS Load Balancer Controller logs
kubectl logs -f deployment/aws-load-balancer-controller -n kube-system
```

### Diagnostic Commands
```bash
# Pod status and events
kubectl get pods -n ag06mixer -o wide
kubectl describe pod <pod-name> -n ag06mixer

# Service and endpoint status
kubectl get svc,endpoints -n ag06mixer

# Resource utilization
kubectl top pods -n ag06mixer
kubectl top nodes

# Recent events
kubectl get events -n ag06mixer --sort-by=.metadata.creationTimestamp
```

## Maintenance

### Rolling Updates
```bash
# Update image version
kubectl set image deployment/ag06mixer-app ag06mixer=ag06mixer:v1.1.0 -n ag06mixer

# Monitor rollout status
kubectl rollout status deployment/ag06mixer-app -n ag06mixer
```

### Rollback Procedures
```bash
# View rollout history
kubectl rollout history deployment/ag06mixer-app -n ag06mixer

# Rollback to previous version
kubectl rollout undo deployment/ag06mixer-app -n ag06mixer
```

### Resource Cleanup
```bash
# Delete specific resources
kubectl delete deployment ag06mixer-app -n ag06mixer

# Delete entire namespace (careful!)
kubectl delete namespace ag06mixer
```

## Best Practices

### Resource Management
- Set appropriate requests and limits
- Use Quality of Service classes effectively
- Monitor resource utilization trends

### Security
- Use least privilege service accounts
- Implement network policies
- Scan container images for vulnerabilities
- Regular security updates

### Observability
- Implement comprehensive health checks
- Use structured logging
- Set up meaningful alerts
- Monitor business metrics

### Deployment
- Use immutable deployments
- Implement proper rollback strategies
- Test deployments in staging first
- Use GitOps for configuration management
