#!/bin/bash
# Production Backup Script for AG06 Mixer

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Database backup
pg_dump ag06_mixer_production | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Application backup
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz /app

# Upload to S3 (if configured)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://$AWS_S3_BUCKET/backups/
    aws s3 cp $BACKUP_DIR/app_backup_$DATE.tar.gz s3://$AWS_S3_BUCKET/backups/
fi

# Clean up old backups (keep 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
