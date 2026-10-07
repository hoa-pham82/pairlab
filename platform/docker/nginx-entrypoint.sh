#!/bin/sh
# Generate self-signed TLS cert and htpasswd, then start nginx.
set -e

mkdir -p /etc/nginx/certs

if [ ! -f /etc/nginx/certs/cert.pem ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/certs/key.pem \
        -out  /etc/nginx/certs/cert.pem \
        -subj "/CN=pairlab-gateway/O=pairlab" 2>/dev/null
fi

HASH=$(openssl passwd -apr1 "${GATEWAY_PASSWORD:-pairlab}")
echo "${GATEWAY_USER:-pairlab}:${HASH}" > /etc/nginx/htpasswd

exec nginx -g "daemon off;"
