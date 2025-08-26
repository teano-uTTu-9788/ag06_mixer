variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
  default     = "ag06mixer.com"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "app_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 3
}

variable "database_instance_type" {
  description = "RDS instance type"
  type        = string
  default     = "db.r5.large"
}

variable "compute_instance_type" {
  description = "EC2 instance type for nodes"
  type        = string
  default     = "t3.medium"
}
