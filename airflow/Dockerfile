FROM apache/airflow:3.3.2-python3.12

COPY airflow/requirements.txt /tmp/credit-risk-airflow-requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.3.2" \
    -r /tmp/credit-risk-airflow-requirements.txt

WORKDIR /opt/airflow/project

COPY --chown=airflow:root airflow ./airflow
COPY --chown=airflow:root config ./config
COPY --chown=airflow:root data/processed ./data/processed
COPY --chown=airflow:root docs ./docs
COPY --chown=airflow:root models ./models
COPY --chown=airflow:root reports ./reports
COPY --chown=airflow:root sql ./sql
COPY --chown=airflow:root src ./src
