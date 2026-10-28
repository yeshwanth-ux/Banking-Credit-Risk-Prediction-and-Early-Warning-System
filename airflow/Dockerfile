FROM apache/airflow:3.3.2-python3.12

COPY airflow/requirements.txt /tmp/credit-risk-airflow-requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.3.2" \
    -r /tmp/credit-risk-airflow-requirements.txt

WORKDIR /opt/airflow/project
