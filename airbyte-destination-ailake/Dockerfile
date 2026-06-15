FROM python:3.11-slim

WORKDIR /app

# Install ailake binary wheel (pre-built) + connector package
COPY pyproject.toml README.md ./
COPY airbyte_destination_ailake/ ./airbyte_destination_ailake/

RUN pip install --no-cache-dir ".[openai,cohere]"

LABEL io.airbyte.version="0.0.1"
LABEL io.airbyte.name="destination-ailake"

ENTRYPOINT ["airbyte-destination-ailake"]
