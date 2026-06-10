# CLOUD_DEPLOY.md — Plugin Deployment Guide

Step-by-step deployment of `spark-plugin`, `trino-plugin`, and `ailake-py` on each cloud.

For storage/catalog configuration see [`INTEGRATIONS.md`](./INTEGRATIONS.md).

---

## Prerequisites — build artifacts

Before deploying to any cloud, build the artifacts locally:

```bash
# Native library (shared by all JVM plugins)
cargo build --release -p ailake-jni
# → target/release/libailake_jni.so

# Spark plugin fat-jar
cd spark-plugin && ./gradlew shadowJar
# → spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar

# Trino plugin fat-jar
cd trino-plugin && ./gradlew shadowJar
# → trino-plugin/build/libs/trino-plugin-0.1.0-plugin.jar

# Python wheel
cd ailake-py && maturin build --release --out dist
# → ailake-py/dist/ailake-*.whl
```

Upload to your cloud artifact store before running the steps below.

---

## 1. AWS

### 1A — Amazon EMR (Spark)

**Upload artifacts to S3:**
```bash
aws s3 cp target/release/libailake_jni.so        s3://my-bucket/ailake/libs/
aws s3 cp spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
                                                  s3://my-bucket/ailake/jars/
aws s3 cp scripts/emr-bootstrap.sh               s3://my-bucket/ailake/bootstrap/
```

**Bootstrap script** (`scripts/emr-bootstrap.sh`):
```bash
#!/bin/bash
set -e
sudo mkdir -p /opt/ailake/lib
sudo aws s3 cp s3://my-bucket/ailake/libs/libailake_jni.so /opt/ailake/lib/
sudo chmod 755 /opt/ailake/lib/libailake_jni.so
```

**Create cluster:**
```bash
aws emr create-cluster \
  --name "ailake-cluster" \
  --release-label emr-7.0.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --bootstrap-actions \
    Path=s3://my-bucket/ailake/bootstrap/emr-bootstrap.sh \
  --configurations '[
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.jars":                          "s3://my-bucket/ailake/jars/spark-plugin-0.1.0-plugin.jar",
        "spark.sql.extensions":                "io.ailake.spark.AilakeSparkExtensions",
        "spark.driver.extraJavaOptions":       "-Djava.library.path=/opt/ailake/lib",
        "spark.executor.extraJavaOptions":     "-Djava.library.path=/opt/ailake/lib"
      }
    }
  ]' \
  --use-default-roles
```

**PySpark job:**
```python
from pyspark.sql import SparkSession
import ailake  # if installed via bootstrap

spark = SparkSession.builder.getOrCreate()

# Vector search via Scala extension
results = spark._jvm.io.ailake.spark.AilakeNative.search(
    "s3://my-lake/docs/",
    query_array,
    10,
)
```

**Scala/Java job submission:**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --class com.example.MyJob \
  s3://my-bucket/jobs/my-job.jar
```

---

### 1B — AWS Glue 4.0 (Spark)

Glue 4.0 runs Spark 3.3 on managed workers. Native `.so` must be downloaded to worker at job start.

**Glue job script** (`glue_job.py`):
```python
import sys, os, subprocess, boto3
from awsglue.utils import getResolvedOptions

# Download native lib at job start
s3 = boto3.client("s3")
os.makedirs("/tmp/ailake/lib", exist_ok=True)
s3.download_file("my-bucket", "ailake/libs/libailake_jni.so", "/tmp/ailake/lib/libailake_jni.so")
os.chmod("/tmp/ailake/lib/libailake_jni.so", 0o755)

# Must set before SparkContext initialises
os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.library.path=/tmp/ailake/lib"

from pyspark.context import SparkContext
from awsglue.context import GlueContext

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
```

**Glue job configuration (console / CloudFormation):**
```yaml
GlueVersion: "4.0"
WorkerType: G.1X
NumberOfWorkers: 10
DefaultArguments:
  --extra-jars: "s3://my-bucket/ailake/jars/spark-plugin-0.1.0-plugin.jar"
  --additional-python-modules: "ailake==0.0.16"
  --conf: "spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions"
```

---

### 1C — AWS Lambda (ailake-py)

**Build Lambda layer:**
```bash
mkdir -p lambda-layer/python
pip install ailake -t lambda-layer/python/
# copy native lib to layer
cp target/release/libailake_jni.so lambda-layer/python/

cd lambda-layer
zip -r ailake-layer.zip python/
aws lambda publish-layer-version \
  --layer-name ailake \
  --zip-file fileb://ailake-layer.zip \
  --compatible-runtimes python3.11 python3.12
```

**Lambda function:**
```python
import ailake
import json, os

# Native lib is in /opt/python/ (layer path)
os.environ["LD_LIBRARY_PATH"] = "/opt/python:" + os.environ.get("LD_LIBRARY_PATH", "")

def handler(event, context):
    results = ailake.search(
        table="s3://my-lake/docs/",
        query=event["embedding"],
        top_k=event.get("top_k", 10),
    )
    return [r.__dict__ for r in results]
```

**Function config:**
```bash
aws lambda create-function \
  --function-name ailake-search \
  --runtime python3.12 \
  --handler handler.handler \
  --layers arn:aws:lambda:us-east-1:123456789:layer:ailake:1 \
  --memory-size 512 \
  --timeout 30 \
  --role arn:aws:iam::123456789:role/lambda-role \
  --zip-file fileb://function.zip
```

---

### 1D — Amazon SageMaker (ailake-py)

**Custom container:**
```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker
RUN pip install ailake
COPY libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib:$LD_LIBRARY_PATH
```

**Processing job:**
```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/ailake-sagemaker:latest",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role="arn:aws:iam::123456789:role/sagemaker-role",
)

processor.run(
    code="scripts/embed_and_ingest.py",
    inputs=[...],
    outputs=[...],
)
```

---

### 1E — Self-hosted Trino on EMR / EC2

```bash
# On each Trino node (coordinator + workers)
sudo mkdir -p /opt/ailake/lib
sudo aws s3 cp s3://my-bucket/ailake/libs/libailake_jni.so /opt/ailake/lib/
sudo chmod 755 /opt/ailake/lib/libailake_jni.so

# Plugin dir
sudo mkdir -p $TRINO_HOME/plugin/ailake
sudo aws s3 cp s3://my-bucket/ailake/jars/trino-plugin-0.1.0-plugin.jar \
              $TRINO_HOME/plugin/ailake/
```

**`etc/jvm.config`:**
```
-Djava.library.path=/opt/ailake/lib
```

**`etc/catalog/ailake.properties`:**
```properties
connector.name=ailake
ailake.table-uri=s3://my-lake/docs/
ailake.vector-column=embedding
ailake.vector-dim=1536
```

Restart Trino. Then:
```sql
SET SESSION ailake.query_vector = '0.021,-0.043,...';
SET SESSION ailake.top_k = 20;
SELECT * FROM ailake.default.search ORDER BY distance;
```

---

## 2. GCP

### 2A — Dataproc (Spark)

**Upload artifacts to GCS:**
```bash
gsutil cp target/release/libailake_jni.so          gs://my-bucket/ailake/libs/
gsutil cp spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
                                                   gs://my-bucket/ailake/jars/
gsutil cp scripts/dataproc-init.sh                 gs://my-bucket/ailake/init/
```

**Init action script** (`scripts/dataproc-init.sh`):
```bash
#!/bin/bash
set -e
mkdir -p /opt/ailake/lib
gsutil cp gs://my-bucket/ailake/libs/libailake_jni.so /opt/ailake/lib/
chmod 755 /opt/ailake/lib/libailake_jni.so
```

**Create cluster:**
```bash
gcloud dataproc clusters create ailake-cluster \
  --region us-central1 \
  --master-machine-type n2-standard-4 \
  --worker-machine-type n2-standard-4 \
  --num-workers 3 \
  --initialization-actions gs://my-bucket/ailake/init/dataproc-init.sh \
  --properties \
    spark:spark.jars=gs://my-bucket/ailake/jars/spark-plugin-0.1.0-plugin.jar,\
    spark:spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions,\
    spark:spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib,\
    spark:spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib
```

**Submit job:**
```bash
gcloud dataproc jobs submit spark \
  --cluster ailake-cluster \
  --region us-central1 \
  --class com.example.MyJob \
  --jars gs://my-bucket/jobs/my-job.jar
```

---

### 2B — Google Cloud Dataflow (ailake-py)

**Custom container (`Dockerfile`):**
```dockerfile
FROM apache/beam_python3.12_sdk:2.59.0
RUN pip install ailake
COPY target/release/libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib:$LD_LIBRARY_PATH
```

**Build and push:**
```bash
docker build -t gcr.io/my-project/beam-ailake:latest .
docker push gcr.io/my-project/beam-ailake:latest
```

**Pipeline with vector search DoFn:**
```python
import apache_beam as beam
import ailake
from apache_beam.options.pipeline_options import PipelineOptions

class VectorSearchFn(beam.DoFn):
    def __init__(self, table_uri):
        self._table_uri = table_uri

    def setup(self):
        self._searcher = ailake.TableSearcher(self._table_uri)

    def process(self, query_embedding):
        results = self._searcher.search(query=query_embedding, top_k=10)
        yield from results

options = PipelineOptions([
    "--runner=DataflowRunner",
    "--project=my-project",
    "--region=us-central1",
    "--temp_location=gs://my-bucket/tmp",
    "--sdk_container_image=gcr.io/my-project/beam-ailake:latest",
    "--experiments=use_runner_v2",
])

with beam.Pipeline(options=options) as p:
    results = (
        p
        | "ReadQueries"  >> beam.io.ReadFromText("gs://my-bucket/queries.txt")
        | "ParseEmbedding" >> beam.Map(lambda x: list(map(float, x.split(","))))
        | "VectorSearch"   >> beam.ParDo(VectorSearchFn("gs://my-lake/docs/"))
        | "WriteResults"   >> beam.io.WriteToText("gs://my-bucket/results/out")
    )
```

**Submit:**
```bash
python pipeline.py \
  --runner DataflowRunner \
  --project my-project \
  --region us-central1 \
  --temp_location gs://my-bucket/tmp \
  --sdk_container_image gcr.io/my-project/beam-ailake:latest
```

---

### 2C — Vertex AI (ailake-py)

**Custom training container:**
```dockerfile
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest
RUN pip install ailake
COPY target/release/libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib:$LD_LIBRARY_PATH
ENTRYPOINT ["python", "train.py"]
```

**Training job:**
```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="ailake-ingest",
    container_uri="gcr.io/my-project/ailake-vertex:latest",
)

job.run(
    machine_type="n1-standard-4",
    replica_count=1,
    args=["--table", "gs://my-lake/docs/", "--top-k", "100"],
)
```

---

## 3. Azure

### 3A — Azure Databricks (Spark)

**Upload artifacts to ADLS Gen2 / DBFS:**
```bash
# Via Databricks CLI
databricks fs cp target/release/libailake_jni.so \
    dbfs:/FileStore/ailake/libs/libailake_jni.so

databricks fs cp spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
    dbfs:/FileStore/ailake/jars/spark-plugin-0.1.0-plugin.jar
```

**Cluster init script** (create in Databricks UI → Compute → Init Scripts):
```bash
#!/bin/bash
mkdir -p /opt/ailake/lib
cp /dbfs/FileStore/ailake/libs/libailake_jni.so /opt/ailake/lib/
chmod 755 /opt/ailake/lib/libailake_jni.so
```

**Cluster configuration (JSON):**
```json
{
  "spark_version": "14.3.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 4,
  "spark_conf": {
    "spark.sql.extensions": "io.ailake.spark.AilakeSparkExtensions",
    "spark.driver.extraJavaOptions": "-Djava.library.path=/opt/ailake/lib",
    "spark.executor.extraJavaOptions": "-Djava.library.path=/opt/ailake/lib"
  },
  "libraries": [
    { "jar": "dbfs:/FileStore/ailake/jars/spark-plugin-0.1.0-plugin.jar" },
    { "pypi": { "package": "ailake==0.0.16" } }
  ],
  "init_scripts": [
    { "dbfs": { "destination": "dbfs:/FileStore/ailake/init/install.sh" } }
  ]
}
```

**Databricks notebook (Scala):**
```scala
import io.ailake.spark.implicits._

val query = Array.fill(1536)(0.0f)  // your embedding

val results = spark.ailakeSearch(
  tableUri    = "abfss://container@account.dfs.core.windows.net/lake/docs/",
  queryVector = query,
  topK        = 20,
)

display(results.orderBy("distance"))
```

**Databricks notebook (Python):**
```python
import ailake
import numpy as np

results = ailake.search(
    table="abfss://container@account.dfs.core.windows.net/lake/docs/",
    query=np.random.rand(1536).astype(np.float32),
    top_k=20,
)
display(results.to_pandas())
```

---

### 3B — Azure HDInsight (Spark)

**Bootstrap action (cluster creation):**
```bash
# script-action.sh
#!/bin/bash
az storage blob download \
  --account-name mystorageaccount \
  --container-name ailake \
  --name libs/libailake_jni.so \
  --file /opt/ailake/lib/libailake_jni.so
chmod 755 /opt/ailake/lib/libailake_jni.so
```

**Create cluster via Azure CLI:**
```bash
az hdinsight create \
  --name my-ailake-cluster \
  --resource-group my-rg \
  --type Spark \
  --version 5.1 \
  --component-version Spark=3.3 \
  --workernode-count 4 \
  --script-actions \
    name=install-ailake \
    uri=https://mystorageaccount.blob.core.windows.net/scripts/script-action.sh \
    roles=headnode,workernode
```

**Spark submit:**
```bash
spark-submit \
  --master yarn \
  --jars wasbs://ailake@mystorageaccount.blob.core.windows.net/jars/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  my-job.jar
```

---

### 3C — Azure Machine Learning (ailake-py)

**Custom environment:**
```yaml
# ailake-env.yml
name: ailake-env
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip:
    - ailake==0.0.16
```

**Or via Docker:**
```dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
RUN pip install ailake
COPY target/release/libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib:$LD_LIBRARY_PATH
```

**AzureML job:**
```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext

ml_client = MLClient.from_config()

env = Environment(
    name="ailake-env",
    build=BuildContext(path="./docker/ailake/"),
)

job = command(
    code="./scripts",
    command="python embed_and_ingest.py --table ${{inputs.table_uri}}",
    inputs={"table_uri": "abfss://container@account.dfs.core.windows.net/lake/docs/"},
    environment=env,
    compute="my-compute-cluster",
)

ml_client.jobs.create_or_update(job)
```

---

## 4. Kubernetes (cloud-agnostic)

For Spark on Kubernetes (EKS / GKE / AKS):

**Spark executor image:**
```dockerfile
FROM apache/spark:3.5.1-scala2.12-java17-python3-ubuntu

USER root
RUN mkdir -p /opt/ailake/lib
COPY target/release/libailake_jni.so /opt/ailake/lib/
ENV LD_LIBRARY_PATH=/opt/ailake/lib:$LD_LIBRARY_PATH
USER spark
```

**spark-submit:**
```bash
spark-submit \
  --master k8s://https://<k8s-api-server>:6443 \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=my-registry/spark-ailake:latest \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --jars s3a://my-bucket/jars/spark-plugin-0.1.0-plugin.jar \
  s3a://my-bucket/jobs/my-job.jar
```

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnsatisfiedLinkError: libailake_jni` | `.so` not on `java.library.path` | Verify init script ran; check `-Djava.library.path` |
| `ailake module not found` | wheel not installed on workers | Add to bootstrap/init script |
| `NotFound` on S3/GCS/ADLS | Missing IAM role / service account | Grant `storage.objectViewer` or `s3:GetObject` |
| Empty search results | Native lib absent (graceful degradation) | Confirm `java.library.path` points to dir with `.so` |
| `OutOfMemoryError` in executor | HNSW loading too many files in parallel | Reduce `search.ef` or add executor memory |
| Plugin jar version mismatch | `spark-plugin` built for Spark 3.5, cluster is 3.3 | Rebuild `spark-plugin` against matching Spark version |
