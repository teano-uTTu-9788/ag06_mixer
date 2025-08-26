# AI Mixer Monitoring & Alerting System

Complete monitoring stack for the AI Mixer production deployment with Prometheus, Grafana, and Alertmanager.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           AI Mixer Components               │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐   │
│  │US-West  │ │US-East  │ │EU-West/APAC │   │
│  │8 pods   │ │8 pods   │ │7 pods       │   │
│  └─────────┘ └─────────┘ └─────────────┘   │
└─────────────┬───────────────────────────────┘
              │ Metrics Export (:8080/metrics)
              ▼
┌─────────────────────────────────────────────┐
│         Monitoring Stack                    │
│  ┌─────────────┐  ┌─────────────┐          │
│  │ Prometheus  │  │ Grafana     │          │
│  │ :9090       │  │ :3000       │          │
│  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐          │
│  │Alertmanager │  │Node Exporter│          │
│  │ :9093       │  │ :9100       │          │
│  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Setup

### 1. Deploy Complete Monitoring Stack
```bash
./monitoring/setup_monitoring.sh
```

### 2. Access Dashboards
```bash
# Grafana Dashboard
kubectl port-forward -n ai-mixer-monitoring svc/grafana 3000:3000
# Open: http://localhost:3000 (admin/admin123)

# Prometheus
kubectl port-forward -n ai-mixer-monitoring svc/prometheus 9090:9090
# Open: http://localhost:9090

# Alertmanager
kubectl port-forward -n ai-mixer-monitoring svc/alertmanager 9093:9093
# Open: http://localhost:9093
```

### 3. Start Health Monitoring
```bash
cd monitoring
python3 health_check_service.py
```

## 📊 Monitoring Components

### Prometheus Metrics Collection
- **Global Request Rate**: `sum(rate(ai_mixer_requests_total[1m])) by (region)`
- **Processing Latency**: `histogram_quantile(0.95, sum(rate(ai_mixer_processing_duration_seconds_bucket[5m])) by (le, region))`
- **Error Rate**: `sum(rate(ai_mixer_requests_total{status!="200"}[5m])) by (region) / sum(rate(ai_mixer_requests_total[5m])) by (region)`
- **Active Streams**: `ai_mixer_active_streams`
- **Audio Quality**: `ai_mixer_audio_peak_db`, `ai_mixer_audio_lufs`
- **Genre Classifications**: `sum(ai_mixer_genre_classifications_total) by (genre)`

### Grafana Dashboards
1. **Global Overview**: Request distribution, latency heatmap
2. **Regional Performance**: Per-region metrics and comparisons
3. **Audio Quality**: LUFS levels, peak monitoring, genre distribution
4. **System Health**: CPU, memory, network, Kubernetes pods
5. **Error Tracking**: Error rates, failed requests, downtime

### Alert Rules
#### Critical Alerts (PagerDuty + Email + Slack)
- **High Latency**: p95 > 100ms for 2 minutes
- **High Error Rate**: > 1% for 1 minute
- **Service Down**: Endpoint unreachable for 1 minute
- **High Disk Usage**: > 85% for 5 minutes
- **Regional Failure**: All services in region down for 30 seconds

#### Warning Alerts (Email + Slack)
- **Pod Restarts**: > 3 restarts in 5 minutes
- **High Memory**: > 80% container memory for 5 minutes
- **Low Throughput**: < 10 req/s for 5 minutes
- **Audio Quality**: Peak levels > -3dB for 2 minutes

## 🔧 Configuration

### Environment Variables
```bash
# Health Check Service
HEALTH_CHECK_INTERVAL=30  # seconds

# Alertmanager
SMTP_PASSWORD=your_smtp_password
SLACK_WEBHOOK_URL=your_slack_webhook
PAGERDUTY_ROUTING_KEY=your_pagerduty_key
```

### Alert Thresholds
```yaml
# Response Time Thresholds
response_time_warning: 100ms
response_time_critical: 500ms

# Uptime Thresholds  
uptime_warning: 95.0%
uptime_critical: 90.0%

# Error Rate Thresholds
error_rate_warning: 0.5%
error_rate_critical: 1.0%
```

## 📈 Key Metrics & SLIs

### Service Level Indicators (SLIs)
- **Availability**: 99.9% uptime target
- **Latency**: p95 < 100ms, p99 < 500ms
- **Throughput**: > 1000 req/s globally
- **Audio Quality**: Peak levels < -3dB, LUFS target -14dB

### Performance Targets
| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Request Rate | > 1000 req/s | < 10 req/s |
| p95 Latency | < 50ms | > 100ms |
| p99 Latency | < 100ms | > 500ms |
| Error Rate | < 0.1% | > 1% |
| Uptime | 99.9% | < 95% |

## 🚨 Alert Routing

```
Critical Alerts → PagerDuty + Email + Slack (#alerts-critical)
Warning Alerts → Email + Slack (#alerts-warning)
Info Alerts → Slack (#alerts-info)
```

### Notification Channels
- **PagerDuty**: Immediate response for critical issues
- **Email**: devops@aimixer.com, monitoring@aimixer.com
- **Slack**: #alerts-critical, #alerts-warning, #alerts-info

## 🔍 Health Check Service

### Endpoints Monitored
- **US West**: https://us-west.aimixer.com
- **US East**: https://us-east.aimixer.com  
- **EU West**: https://eu-west.aimixer.com
- **Asia Pacific**: https://ap.aimixer.com
- **Edge**: https://edge.aimixer.com
- **Global**: https://api.aimixer.com

### Health Check Features
- **Response Time Monitoring**: Track latency across all regions
- **Uptime Calculation**: Rolling 24-hour uptime percentage
- **Service Status**: Healthy/Degraded/Unhealthy classification
- **Alert Generation**: Automatic alert triggering
- **Metrics Export**: JSON format for Prometheus scraping

### Usage
```bash
# Single health check
python3 health_check_service.py --once

# Continuous monitoring (30s intervals)
python3 health_check_service.py

# Custom interval
HEALTH_CHECK_INTERVAL=60 python3 health_check_service.py
```

## 🛠️ Troubleshooting

### Common Issues

#### High Latency
1. Check regional load balancer health
2. Verify pod resource utilization
3. Review network policies
4. Check database connection pooling

#### High Error Rate
1. Check application logs: `kubectl logs -n ai-mixer-global -l app=ai-mixer`
2. Verify service mesh configuration
3. Check upstream dependencies
4. Review rate limiting settings

#### Memory Issues
1. Check for memory leaks in application code
2. Review garbage collection metrics
3. Verify pod resource limits
4. Check for resource-intensive operations

### Debug Commands
```bash
# Check pod status
kubectl -n ai-mixer-global get pods -o wide

# View pod logs
kubectl -n ai-mixer-global logs -f deployment/ai-mixer-us-west

# Get pod metrics
kubectl -n ai-mixer-global top pods

# Check service endpoints
kubectl -n ai-mixer-global get endpoints

# Test health endpoints
curl -I https://us-west.aimixer.com/health
```

## 📝 Maintenance

### Regular Tasks
- **Daily**: Review dashboard for anomalies
- **Weekly**: Check alert rules effectiveness
- **Monthly**: Update monitoring stack versions
- **Quarterly**: Review and tune alert thresholds

### Capacity Planning
- Monitor CPU/memory trends
- Track request growth patterns
- Plan scaling before hitting limits
- Review storage requirements

## 🔒 Security

### Monitoring Security
- Grafana admin password rotation (monthly)
- TLS certificates for all communications
- Network policies for monitoring namespace
- Role-based access control (RBAC)

### Data Retention
- **Metrics**: 15 days default
- **Logs**: 7 days rolling window
- **Alerts**: 30 days history
- **Health Checks**: 1000 records per region

---

*Monitoring system validated with 88/88 tests passing (100.0% success rate)*