# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "ag06mixer-cluster"
  cluster_version = "1.28"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      name = "ag06mixer-node-group"

      instance_types = ["t3.medium"]

      min_size     = 2
      max_size     = 10
      desired_size = 3

      vpc_security_group_ids = [aws_security_group.app.id]
    }
  }

  tags = {
    Name = "ag06mixer-eks-cluster"
  }
}
