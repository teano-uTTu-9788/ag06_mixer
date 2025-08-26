# Infrastructure Deployment Guide

## AWS Infrastructure Components

### 1. Virtual Private Cloud (VPC)
- **CIDR**: 10.0.0.0/16
- **Public Subnets**: 10.0.101.0/24, 10.0.102.0/24, 10.0.103.0/24
- **Private Subnets**: 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24
- **Availability Zones**: 3 AZs for high availability

### 2. Elastic Kubernetes Service (EKS)
- **Cluster Version**: 1.28
- **Node Groups**: Managed node groups with auto-scaling
- **Instance Type**: t3.medium
- **Scaling**: 2-10 nodes

### 3. Relational Database Service (RDS)
- **Engine**: PostgreSQL 14.9
- **Instance Class**: db.r5.large
- **Multi-AZ**: Enabled for high availability
- **Backup**: 30-day retention period
- **Encryption**: Enabled with AWS KMS

### 4. Application Load Balancer (ALB)
- **Type**: Application Load Balancer
- **Scheme**: Internet-facing
- **SSL**: AWS Certificate Manager integration
- **Health Checks**: Application health endpoint

### 5. Route 53 DNS
- **Hosted Zone**: ag06mixer.com
- **Records**: A records for primary and API domains
- **SSL Validation**: DNS validation method

## Terraform Modules

### VPC Module
- Uses terraform-aws-modules/vpc/aws
- Configures subnets, route tables, and NAT gateways
- Enables DNS hostnames and resolution

### EKS Module  
- Uses terraform-aws-modules/eks/aws
- Configures cluster and managed node groups
- Sets up IAM roles and security groups

## Security Configuration

### IAM Roles
- EKS Cluster Service Role
- EKS Node Group Instance Role
- AWS Load Balancer Controller Role
- Application Pod Service Account Role

### Security Groups
- Application security group (ports 80, 443)
- Database security group (port 5432 from app)
- EKS cluster security groups (managed by module)

### Network ACLs
- Default VPC network ACLs
- Subnet-level traffic filtering
- Egress and ingress rule definitions

## Deployment Process

### Pre-deployment Checklist
- [ ] AWS CLI configured with appropriate credentials
- [ ] Terraform installed (>= 1.0)
- [ ] Domain registered or available for registration
- [ ] S3 bucket created for Terraform state

### Deployment Steps

1. **Initialize Terraform**
   ```bash
   cd infrastructure
   terraform init
   ```

2. **Review Plan**
   ```bash
   terraform plan
   ```

3. **Deploy Infrastructure**
   ```bash
   terraform apply
   ```

4. **Verify Deployment**
   ```bash
   terraform output
   ```

### Post-deployment Tasks
- Configure kubectl with EKS cluster
- Install AWS Load Balancer Controller
- Set up monitoring and logging
- Configure application secrets

## Monitoring Infrastructure

### CloudWatch Integration
- EKS cluster logging enabled
- VPC Flow Logs for network monitoring
- RDS Performance Insights
- ALB access logs

### Cost Monitoring
- AWS Cost Explorer integration
- Resource tagging for cost allocation
- Budget alerts for spending thresholds

## Backup and Recovery

### Database Backups
- Automated daily backups
- Point-in-time recovery (PITR)
- Cross-region backup replication
- Backup retention policies

### Infrastructure Recovery
- Terraform state backup
- Infrastructure as Code versioning
- Disaster recovery procedures
- Multi-region deployment capability

## Scaling and Performance

### Auto-Scaling Configuration
- EKS Cluster Autoscaler
- Horizontal Pod Autoscaler
- Application Load Balancer health checks
- Database read replica auto-scaling

### Performance Optimization
- EBS-optimized instances
- GP3 storage for better IOPS
- Enhanced networking enabled
- Placement groups for performance

## Troubleshooting

### Common Infrastructure Issues

1. **VPC Configuration**
   - Route table misconfigurations
   - Security group rules
   - NAT gateway connectivity

2. **EKS Cluster Issues**
   - Node group scaling problems
   - IAM permission errors
   - Network connectivity issues

3. **Database Connectivity**
   - Security group rules
   - Subnet group configuration
   - Parameter group settings

4. **Load Balancer Issues**
   - Target group health checks
   - SSL certificate validation
   - DNS resolution problems

### Diagnostic Commands
```bash
# Check VPC configuration
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=ag06mixer-vpc"

# Verify EKS cluster status
aws eks describe-cluster --name ag06mixer-cluster

# Check RDS instance
aws rds describe-db-instances --db-instance-identifier ag06mixer-db

# Test ALB health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>
```

## Maintenance

### Regular Maintenance Tasks
- Update Terraform modules
- Patch EKS cluster version
- Update RDS minor versions
- Review and rotate credentials
- Monitor cost optimization opportunities

### Security Updates
- Apply security patches
- Update IAM policies
- Review security group rules
- Audit access logs
- Vulnerability assessments

## Best Practices

### Infrastructure as Code
- Version control all Terraform code
- Use consistent naming conventions
- Implement proper state management
- Regular state file backups

### Security
- Principle of least privilege
- Enable encryption at rest and in transit
- Regular security audits
- Multi-factor authentication
- Network segmentation

### Cost Management
- Right-size resources
- Use Reserved Instances for predictable workloads
- Implement auto-scaling policies
- Regular cost reviews
- Resource cleanup procedures
