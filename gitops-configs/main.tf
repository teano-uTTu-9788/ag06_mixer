
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  backend "gcs" {
    bucket = "enterprise-terraform-state"
    prefix = "production/state"
  }
}

# Google GKE Cluster
module "gke_cluster" {
  source = "./modules/gke"
  
  project_id     = var.google_project_id
  region         = "us-central1"
  cluster_name   = "enterprise-production-gke"
  
  node_pools = [
    {
      name               = "default-pool"
      machine_type       = "n2-standard-8"
      min_count          = 3
      max_count          = 20
      disk_size_gb       = 100
      disk_type          = "pd-ssd"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = false
      enable_secure_boot = true
    }
  ]
  
  features = {
    workload_identity = true
    binary_authorization = true
    shielded_nodes = true
    confidential_nodes = true
    network_policy = true
    vertical_pod_autoscaling = true
    cluster_autoscaling = true
  }
}

# Azure AKS Cluster
module "aks_cluster" {
  source = "./modules/aks"
  
  resource_group_name = azurerm_resource_group.main.name
  location           = "westus2"
  cluster_name       = "enterprise-production-aks"
  
  default_node_pool = {
    name                = "system"
    vm_size            = "Standard_DS4_v2"
    node_count         = 3
    enable_auto_scaling = true
    min_count          = 3
    max_count          = 10
  }
  
  features = {
    azure_policy_enabled = true
    azure_ad_rbac_enabled = true
    defender_enabled = true
    key_vault_secrets_provider_enabled = true
    workload_identity_enabled = true
  }
}

# AWS EKS Cluster
module "eks_cluster" {
  source = "./modules/eks"
  
  cluster_name    = "enterprise-production-eks"
  cluster_version = "1.28"
  region         = "us-west-2"
  
  node_groups = {
    main = {
      instance_types = ["m5.2xlarge"]
      min_size      = 3
      max_size      = 20
      desired_size  = 5
      
      labels = {
        Environment = "production"
        Tier        = "backend"
      }
    }
  }
  
  features = {
    enable_irsa = true
    enable_ssm = true
    enable_cluster_autoscaler = true
    enable_metrics_server = true
    enable_aws_load_balancer_controller = true
  }
}

# Anthos Config Management for all clusters
resource "google_gke_hub_membership" "clusters" {
  for_each = {
    gke = module.gke_cluster.cluster_endpoint
    aks = module.aks_cluster.cluster_endpoint
    eks = module.eks_cluster.cluster_endpoint
  }
  
  membership_id = each.key
  endpoint {
    gke_cluster {
      resource_link = each.value
    }
  }
}

# Output endpoints
output "cluster_endpoints" {
  value = {
    gke = module.gke_cluster.endpoint
    aks = module.aks_cluster.endpoint
    eks = module.eks_cluster.endpoint
  }
  sensitive = true
}
