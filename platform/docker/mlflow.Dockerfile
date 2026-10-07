FROM python:3.11-slim

# Install mlflow matching the project venv version, plus Postgres and S3 deps
RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.131

EXPOSE 5000
ENTRYPOINT ["mlflow"]
