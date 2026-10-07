FROM nginx:1.27-alpine
RUN apk add --no-cache openssl
COPY nginx-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Remove the default nginx page
RUN rm -f /etc/nginx/conf.d/default.conf.default
ENTRYPOINT ["/entrypoint.sh"]
