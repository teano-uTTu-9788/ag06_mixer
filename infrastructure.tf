
# Terraform configuration for AiOke 2025 Ultimate

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Multi-region deployment
variable "regions" {
  default = {
    gcp     = ["us-central1", "europe-west1"]
    aws     = ["us-east-1", "eu-west-1"]
    azure   = ["eastus", "westeurope"]
  }
}

# GKE Cluster for Google Vertex AI
resource "google_container_cluster" "aioke_gke" {
  for_each = toset(var.regions.gcp)
  
  name     = "aioke-2025-${each.key}"
  location = each.key
  
  initial_node_count = 3
  
  node_config {
    machine_type = "n2-standard-8"
    disk_size_gb = 100
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = 3
      maximum       = 100
    }
    resource_limits {
      resource_type = "memory"
      minimum       = 12
      maximum       = 400
    }
  }
}

# EKS Cluster for AWS
resource "aws_eks_cluster" "aioke_eks" {
  for_each = toset(var.regions.aws)
  
  name     = "aioke-2025-${each.key}"
  role_arn = aws_iam_role.eks_cluster.arn
  
  vpc_config {
    subnet_ids = aws_subnet.eks[*].id
  }
}

# AKS Cluster for Azure
resource "azurerm_kubernetes_cluster" "aioke_aks" {
  for_each = toset(var.regions.azure)
  
  name                = "aioke-2025-${each.key}"
  location            = each.key
  resource_group_name = azurerm_resource_group.aioke.name
  dns_prefix          = "aioke2025"
  
  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D8s_v3"
    
    enable_auto_scaling = true
    min_count          = 3
    max_count          = 20
  }
}

# Global Load Balancer
resource "google_compute_global_forwarding_rule" "aioke_global_lb" {
  name       = "aioke-2025-global-lb"
  target     = google_compute_target_https_proxy.aioke.id
  port_range = "443"
}

output "endpoints" {
  value = {
    gcp   = google_compute_global_forwarding_rule.aioke_global_lb.ip_address
    aws   = aws_lb.aioke.dns_name
    azure = azurerm_public_ip.aioke.ip_address
  }
}
