FROM apache/airflow:2.9.3-python3.11

# LightGBM requires libgomp1 (OpenMP runtime)
USER root
RUN apt-get update -qq && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Separate venv for ML packages to avoid fighting Airflow's pinned SQLAlchemy/pandas
RUN python -m venv /opt/ml-venv

RUN /opt/ml-venv/bin/pip install --no-cache-dir \
    "lightgbm==4.7.0" \
    "scikit-learn==1.9.1" \
    "mlflow==3.16.1" \
    "deltalake==1.6.6" \
    "feast[postgres]==0.66.0" \
    "redis>=4.2,<8" \
    "psycopg2-binary==2.9.9" \
    "psycopg[binary,pool]>=3.1" \
    "prometheus-client>=0.20" \
    "pandas>=2.2" \
    "pyarrow>=16" \
    "joblib>=1.4"

USER airflow
