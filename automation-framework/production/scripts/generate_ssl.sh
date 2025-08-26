#!/bin/bash
# SSL Certificate Generation for ag06mixer.com

# Generate Let's Encrypt certificate
certbot certonly --nginx -d ag06mixer.com -d www.ag06mixer.com --non-interactive --agree-tos --email admin@ag06mixer.com

# Setup auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -

# Configure nginx SSL
cp /etc/letsencrypt/live/ag06mixer.com/fullchain.pem /etc/ssl/certs/
cp /etc/letsencrypt/live/ag06mixer.com/privkey.pem /etc/ssl/private/
