# Database subnet group
resource "aws_db_subnet_group" "main" {
  name       = "ag06mixer-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "ag06mixer-db-subnet-group"
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "ag06mixer-db"

  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.r5.large"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true

  db_name  = "ag06mixer"
  username = "ag06mixer_admin"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 30
  backup_window          = "07:00-09:00"
  maintenance_window     = "sun:09:00-sun:10:00"

  multi_az               = true
  publicly_accessible    = false
  copy_tags_to_snapshot  = true
  deletion_protection    = true

  skip_final_snapshot = false
  final_snapshot_identifier = "ag06mixer-db-final-snapshot"

  tags = {
    Name = "ag06mixer-production-db"
  }
}

# Database password
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name = "ag06mixer/database/credentials"
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = random_password.db_password.result
    endpoint = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
  })
}
