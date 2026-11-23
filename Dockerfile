# Airflow image with the project's ETL + CPU inference deps baked in (faster startups than _PIP_ADDITIONAL_REQUIREMENTS).
FROM apache/airflow:2.10.5-python3.11
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
# pinning apache-airflow stops pip from silently upgrading Airflow itself
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/requirements-airflow.txt
