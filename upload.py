import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

from modiml.logging.mlflow_logger import MLflowLogger

results = '/home/ubuntu/results.zip'
params = {'model name': 'rohit-26',}
headers = ['none']

mlflow_logger = MLflowLogger(
        "Neural Talk Head - 2",
        'rohit-26-results',
        params=params,
        metrics_headers=headers,
)
load_dotenv(".env")

mlflow.log_artifact(results, "Results")
