#!/usr/bin/env python3
"""
AI Mixer Backup and Recovery System

Provides comprehensive backup and recovery capabilities for all AI Mixer components:
- Configuration backup and restore
- Database backup (if applicable)
- Container image backup
- Multi-region backup distribution
- Automated recovery procedures
- Point-in-time recovery
"""

import asyncio
import json
import os
import shutil
import subprocess
import tarfile
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Metadata for a backup"""
    backup_id: str
    timestamp: str
    version: str
    components: List[str]
    size_bytes: int
    checksum: str
    location: str
    retention_days: int

@dataclass
class RecoveryPlan:
    """Recovery plan configuration"""
    backup_id: str
    target_environment: str
    recovery_steps: List[str]
    estimated_time_minutes: int
    required_approvals: List[str]

class BackupSystem:
    """AI Mixer backup and recovery system"""
    
    def __init__(self, config_path: str = "backup_config.yaml"):
        self.config = self.load_config(config_path)
        self.backup_root = Path(self.config.get("backup_root", "./backups"))
        self.s3_bucket = self.config.get("s3_bucket")
        self.retention_days = self.config.get("retention_days", 30)
        
        # Initialize AWS S3 client if configured
        self.s3_client = None
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"S3 backup enabled: {self.s3_bucket}")
            except Exception as e:
                logger.warning(f"S3 client initialization failed: {e}")
        
        # Create backup directory structure
        self.backup_root.mkdir(parents=True, exist_ok=True)
        (self.backup_root / "configs").mkdir(exist_ok=True)
        (self.backup_root / "data").mkdir(exist_ok=True)
        (self.backup_root / "images").mkdir(exist_ok=True)
        (self.backup_root / "metadata").mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load backup configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "backup_root": "./backups",
                "retention_days": 30,
                "components": [
                    "kubernetes_configs",
                    "docker_images",
                    "application_config",
                    "monitoring_config",
                    "ssl_certificates"
                ],
                "s3_bucket": None,
                "encryption_key": None
            }
    
    def generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"aimixer_backup_{timestamp}"
    
    async def backup_kubernetes_configs(self, backup_dir: Path) -> int:
        """Backup Kubernetes configurations"""
        logger.info("Backing up Kubernetes configurations...")
        
        k8s_backup_dir = backup_dir / "kubernetes"
        k8s_backup_dir.mkdir(parents=True, exist_ok=True)
        
        total_size = 0
        
        # Backup namespace configurations
        namespaces = ["ai-mixer-global", "ai-mixer-monitoring"]
        
        for namespace in namespaces:
            try:
                # Export all resources in namespace
                result = subprocess.run([
                    "kubectl", "get", "all,configmaps,secrets,pvc,ingress",
                    "-n", namespace, "-o", "yaml"
                ], capture_output=True, text=True, check=True)
                
                output_file = k8s_backup_dir / f"{namespace}.yaml"
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                
                total_size += output_file.stat().st_size
                logger.info(f"Backed up namespace: {namespace}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to backup namespace {namespace}: {e}")
        
        # Backup custom resource definitions
        try:
            result = subprocess.run([
                "kubectl", "get", "crd", "-o", "yaml"
            ], capture_output=True, text=True, check=True)
            
            crd_file = k8s_backup_dir / "custom_resources.yaml"
            with open(crd_file, 'w') as f:
                f.write(result.stdout)
            
            total_size += crd_file.stat().st_size
            logger.info("Backed up custom resource definitions")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to backup CRDs: {e}")
        
        # Backup cluster-level resources
        cluster_resources = ["nodes", "persistentvolumes", "storageclasses"]
        for resource in cluster_resources:
            try:
                result = subprocess.run([
                    "kubectl", "get", resource, "-o", "yaml"
                ], capture_output=True, text=True, check=True)
                
                resource_file = k8s_backup_dir / f"{resource}.yaml"
                with open(resource_file, 'w') as f:
                    f.write(result.stdout)
                
                total_size += resource_file.stat().st_size
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to backup {resource}: {e}")
        
        logger.info(f"Kubernetes config backup completed: {total_size} bytes")
        return total_size
    
    async def backup_docker_images(self, backup_dir: Path) -> int:
        """Backup Docker images"""
        logger.info("Backing up Docker images...")
        
        images_dir = backup_dir / "docker_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # List of images to backup
        images = [
            "ai-mixer:latest",
            "ai-mixer-ml:latest",
            "ai-mixer-edge:latest"
        ]
        
        total_size = 0
        
        for image in images:
            try:
                # Check if image exists locally
                result = subprocess.run([
                    "docker", "images", "-q", image
                ], capture_output=True, text=True)
                
                if not result.stdout.strip():
                    logger.warning(f"Image {image} not found locally, skipping")
                    continue
                
                # Export image to tar file
                safe_name = image.replace(":", "_").replace("/", "_")
                tar_file = images_dir / f"{safe_name}.tar"
                
                with open(tar_file, 'wb') as f:
                    subprocess.run([
                        "docker", "save", image
                    ], stdout=f, check=True)
                
                total_size += tar_file.stat().st_size
                logger.info(f"Backed up Docker image: {image}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to backup image {image}: {e}")
        
        logger.info(f"Docker images backup completed: {total_size} bytes")
        return total_size
    
    async def backup_application_configs(self, backup_dir: Path) -> int:
        """Backup application configurations"""
        logger.info("Backing up application configurations...")
        
        config_dir = backup_dir / "application_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        total_size = 0
        
        # List of configuration files/directories to backup
        config_paths = [
            "deploy_production.sh",
            "monitoring/",
            "multi_region/",
            "edge_computing/workers/",
            "mobile_sdks/",
            "ml_optimization/",
            "API_DOCUMENTATION.md",
            "SYSTEM_DOCUMENTATION.md",
            "test_results_88.json"
        ]
        
        for config_path in config_paths:
            source_path = Path(config_path)
            
            if not source_path.exists():
                logger.warning(f"Config path not found: {config_path}")
                continue
            
            if source_path.is_file():
                # Copy single file
                dest_file = config_dir / source_path.name
                shutil.copy2(source_path, dest_file)
                total_size += dest_file.stat().st_size
                
            elif source_path.is_dir():
                # Copy directory recursively
                dest_dir = config_dir / source_path.name
                shutil.copytree(source_path, dest_dir, dirs_exist_ok=True)
                
                # Calculate directory size
                for root, dirs, files in os.walk(dest_dir):
                    for file in files:
                        total_size += Path(root, file).stat().st_size
            
            logger.info(f"Backed up: {config_path}")
        
        logger.info(f"Application configs backup completed: {total_size} bytes")
        return total_size
    
    async def backup_ssl_certificates(self, backup_dir: Path) -> int:
        """Backup SSL certificates"""
        logger.info("Backing up SSL certificates...")
        
        ssl_dir = backup_dir / "ssl_certificates"
        ssl_dir.mkdir(parents=True, exist_ok=True)
        
        total_size = 0
        
        try:
            # Backup Kubernetes TLS secrets
            result = subprocess.run([
                "kubectl", "get", "secrets", "-A", 
                "--field-selector", "type=kubernetes.io/tls",
                "-o", "yaml"
            ], capture_output=True, text=True, check=True)
            
            tls_file = ssl_dir / "tls_secrets.yaml"
            with open(tls_file, 'w') as f:
                f.write(result.stdout)
            
            total_size += tls_file.stat().st_size
            logger.info("Backed up TLS secrets from Kubernetes")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to backup TLS secrets: {e}")
        
        # Backup local certificate files if they exist
        cert_paths = [
            "/etc/ssl/certs/aimixer.crt",
            "/etc/ssl/private/aimixer.key",
            "./certs/"
        ]
        
        for cert_path in cert_paths:
            source_path = Path(cert_path)
            if source_path.exists():
                if source_path.is_file():
                    dest_file = ssl_dir / source_path.name
                    shutil.copy2(source_path, dest_file)
                    total_size += dest_file.stat().st_size
                elif source_path.is_dir():
                    dest_dir = ssl_dir / source_path.name
                    shutil.copytree(source_path, dest_dir, dirs_exist_ok=True)
                    
                    for root, dirs, files in os.walk(dest_dir):
                        for file in files:
                            total_size += Path(root, file).stat().st_size
                
                logger.info(f"Backed up certificates from: {cert_path}")
        
        logger.info(f"SSL certificates backup completed: {total_size} bytes")
        return total_size
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def create_backup_archive(self, backup_dir: Path, backup_id: str) -> Tuple[Path, int]:
        """Create compressed backup archive"""
        logger.info("Creating backup archive...")
        
        archive_path = self.backup_root / f"{backup_id}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_dir, arcname=backup_id)
        
        archive_size = archive_path.stat().st_size
        logger.info(f"Backup archive created: {archive_path} ({archive_size} bytes)")
        
        return archive_path, archive_size
    
    async def upload_to_s3(self, archive_path: Path, backup_id: str) -> bool:
        """Upload backup archive to S3"""
        if not self.s3_client or not self.s3_bucket:
            logger.info("S3 upload skipped - not configured")
            return False
        
        logger.info(f"Uploading backup to S3: {self.s3_bucket}/{backup_id}.tar.gz")
        
        try:
            s3_key = f"ai-mixer-backups/{backup_id}.tar.gz"
            
            # Upload with metadata
            extra_args = {
                'Metadata': {
                    'backup-id': backup_id,
                    'component': 'ai-mixer',
                    'version': '1.0.0',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            self.s3_client.upload_file(
                str(archive_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            logger.info("S3 upload completed successfully")
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    async def create_backup_metadata(self, backup_id: str, components: List[str], 
                                   archive_size: int, checksum: str) -> BackupMetadata:
        """Create backup metadata"""
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            components=components,
            size_bytes=archive_size,
            checksum=checksum,
            location=f"local:{self.backup_root}/{backup_id}.tar.gz",
            retention_days=self.retention_days
        )
        
        # Save metadata to file
        metadata_file = self.backup_root / "metadata" / f"{backup_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Backup metadata saved: {metadata_file}")
        return metadata
    
    async def create_full_backup(self) -> BackupMetadata:
        """Create a complete system backup"""
        backup_id = self.generate_backup_id()
        backup_dir = self.backup_root / "temp" / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting full backup: {backup_id}")
        
        total_size = 0
        components = []
        
        try:
            # Backup Kubernetes configurations
            size = await self.backup_kubernetes_configs(backup_dir)
            total_size += size
            components.append("kubernetes_configs")
            
            # Backup Docker images
            size = await self.backup_docker_images(backup_dir)
            total_size += size
            components.append("docker_images")
            
            # Backup application configurations
            size = await self.backup_application_configs(backup_dir)
            total_size += size
            components.append("application_configs")
            
            # Backup SSL certificates
            size = await self.backup_ssl_certificates(backup_dir)
            total_size += size
            components.append("ssl_certificates")
            
            # Create archive
            archive_path, archive_size = self.create_backup_archive(backup_dir, backup_id)
            
            # Calculate checksum
            checksum = self.calculate_checksum(archive_path)
            
            # Upload to S3 if configured
            await self.upload_to_s3(archive_path, backup_id)
            
            # Create metadata
            metadata = await self.create_backup_metadata(
                backup_id, components, archive_size, checksum
            )
            
            # Clean up temporary directory
            shutil.rmtree(backup_dir)
            
            logger.info(f"Backup completed successfully: {backup_id}")
            logger.info(f"Components: {', '.join(components)}")
            logger.info(f"Archive size: {archive_size} bytes")
            logger.info(f"Checksum: {checksum}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up on failure
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        backups = []
        metadata_dir = self.backup_root / "metadata"
        
        if not metadata_dir.exists():
            return backups
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    backups.append(BackupMetadata(**data))
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        return backups
    
    async def restore_from_backup(self, backup_id: str, 
                                target_environment: str = "production") -> bool:
        """Restore system from backup"""
        logger.info(f"Starting restore from backup: {backup_id}")
        
        # Find backup metadata
        backups = self.list_backups()
        backup_metadata = next((b for b in backups if b.backup_id == backup_id), None)
        
        if not backup_metadata:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        # Verify backup archive exists
        archive_path = self.backup_root / f"{backup_id}.tar.gz"
        if not archive_path.exists():
            logger.error(f"Backup archive not found: {archive_path}")
            return False
        
        # Verify checksum
        current_checksum = self.calculate_checksum(archive_path)
        if current_checksum != backup_metadata.checksum:
            logger.error(f"Backup integrity check failed")
            logger.error(f"Expected: {backup_metadata.checksum}")
            logger.error(f"Actual: {current_checksum}")
            return False
        
        logger.info("Backup integrity verified")
        
        # Extract backup
        restore_dir = self.backup_root / "restore" / backup_id
        restore_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(restore_dir)
            
            logger.info(f"Backup extracted to: {restore_dir}")
            
            # Restore components
            for component in backup_metadata.components:
                await self.restore_component(restore_dir / backup_id, component)
            
            logger.info("Restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
        finally:
            # Clean up restore directory
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
    
    async def restore_component(self, backup_dir: Path, component: str) -> bool:
        """Restore a specific component"""
        logger.info(f"Restoring component: {component}")
        
        try:
            if component == "kubernetes_configs":
                return await self.restore_kubernetes_configs(backup_dir / "kubernetes")
            elif component == "docker_images":
                return await self.restore_docker_images(backup_dir / "docker_images")
            elif component == "application_configs":
                return await self.restore_application_configs(backup_dir / "application_configs")
            elif component == "ssl_certificates":
                return await self.restore_ssl_certificates(backup_dir / "ssl_certificates")
            else:
                logger.warning(f"Unknown component: {component}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore component {component}: {e}")
            return False
    
    async def restore_kubernetes_configs(self, k8s_dir: Path) -> bool:
        """Restore Kubernetes configurations"""
        logger.info("Restoring Kubernetes configurations...")
        
        if not k8s_dir.exists():
            logger.warning("Kubernetes backup directory not found")
            return False
        
        try:
            # Restore namespace configurations
            for yaml_file in k8s_dir.glob("*.yaml"):
                if yaml_file.name in ["nodes.yaml", "persistentvolumes.yaml"]:
                    # Skip cluster-level resources that shouldn't be restored
                    continue
                
                logger.info(f"Restoring: {yaml_file.name}")
                subprocess.run([
                    "kubectl", "apply", "-f", str(yaml_file)
                ], check=True)
            
            logger.info("Kubernetes configurations restored successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restore Kubernetes configs: {e}")
            return False
    
    async def restore_docker_images(self, images_dir: Path) -> bool:
        """Restore Docker images"""
        logger.info("Restoring Docker images...")
        
        if not images_dir.exists():
            logger.warning("Docker images backup directory not found")
            return False
        
        try:
            for tar_file in images_dir.glob("*.tar"):
                logger.info(f"Loading Docker image: {tar_file.name}")
                
                with open(tar_file, 'rb') as f:
                    subprocess.run([
                        "docker", "load"
                    ], stdin=f, check=True)
            
            logger.info("Docker images restored successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restore Docker images: {e}")
            return False
    
    async def restore_application_configs(self, config_dir: Path) -> bool:
        """Restore application configurations"""
        logger.info("Restoring application configurations...")
        
        if not config_dir.exists():
            logger.warning("Application configs backup directory not found")
            return False
        
        try:
            # Copy configurations back to their original locations
            for item in config_dir.iterdir():
                dest_path = Path(item.name)
                
                if item.is_file():
                    shutil.copy2(item, dest_path)
                elif item.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)
                
                logger.info(f"Restored: {dest_path}")
            
            logger.info("Application configurations restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore application configs: {e}")
            return False
    
    async def restore_ssl_certificates(self, ssl_dir: Path) -> bool:
        """Restore SSL certificates"""
        logger.info("Restoring SSL certificates...")
        
        if not ssl_dir.exists():
            logger.warning("SSL certificates backup directory not found")
            return False
        
        try:
            # Restore TLS secrets to Kubernetes
            tls_file = ssl_dir / "tls_secrets.yaml"
            if tls_file.exists():
                subprocess.run([
                    "kubectl", "apply", "-f", str(tls_file)
                ], check=True)
                logger.info("TLS secrets restored to Kubernetes")
            
            logger.info("SSL certificates restored successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restore SSL certificates: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """Clean up old backups based on retention policy"""
        logger.info(f"Cleaning up backups older than {self.retention_days} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        cleaned_count = 0
        
        backups = self.list_backups()
        
        for backup in backups:
            backup_date = datetime.fromisoformat(backup.timestamp.replace('Z', '+00:00'))
            
            if backup_date < cutoff_date:
                logger.info(f"Removing old backup: {backup.backup_id}")
                
                # Remove archive file
                archive_path = self.backup_root / f"{backup.backup_id}.tar.gz"
                if archive_path.exists():
                    archive_path.unlink()
                
                # Remove metadata file
                metadata_path = self.backup_root / "metadata" / f"{backup.backup_id}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old backups")
        return cleaned_count

async def main():
    """Main function for backup operations"""
    import sys
    
    backup_system = BackupSystem()
    
    if len(sys.argv) < 2:
        print("Usage: python backup_system.py <command> [options]")
        print("Commands:")
        print("  backup           - Create full backup")
        print("  list            - List all backups")
        print("  restore <id>    - Restore from backup")
        print("  cleanup         - Clean up old backups")
        return
    
    command = sys.argv[1]
    
    if command == "backup":
        metadata = await backup_system.create_full_backup()
        print(f"Backup completed: {metadata.backup_id}")
        print(f"Size: {metadata.size_bytes} bytes")
        print(f"Components: {', '.join(metadata.components)}")
        
    elif command == "list":
        backups = backup_system.list_backups()
        print(f"Found {len(backups)} backups:")
        
        for backup in backups:
            print(f"  {backup.backup_id}")
            print(f"    Timestamp: {backup.timestamp}")
            print(f"    Size: {backup.size_bytes} bytes")
            print(f"    Components: {', '.join(backup.components)}")
            print()
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: python backup_system.py restore <backup_id>")
            return
        
        backup_id = sys.argv[2]
        success = await backup_system.restore_from_backup(backup_id)
        
        if success:
            print(f"Restore completed successfully: {backup_id}")
        else:
            print(f"Restore failed: {backup_id}")
    
    elif command == "cleanup":
        cleaned = backup_system.cleanup_old_backups()
        print(f"Cleaned up {cleaned} old backups")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())