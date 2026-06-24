#!/usr/bin/env bash
# Airflow demo entrypoint — init DB, create admin user, start scheduler + webserver.
set -e

airflow db migrate

# Create admin user (no-op if it already exists).
airflow users create \
  --role Admin \
  --username admin \
  --password admin \
  --email admin@ailake.demo \
  --firstname AI \
  --lastname Lake \
  2>/dev/null || true

# Scheduler in background, webserver in foreground.
airflow scheduler &
exec airflow webserver --port 8080
