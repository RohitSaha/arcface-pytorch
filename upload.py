import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

from modiml.logging.mlflow_logger import MLflowLogger

image_dir = '/home/ubuntu/arcface-pytorch/synthesized_images'
descriptor_dir = '/home/ubuntu/arcface-pytorch/descriptors'
landmark_dir = '/home/ubuntu/arcface-pytorch/landmark'

CELEB_IDX = 'id00017' 
mlflow_params = {"celeb_id": CELEB_IDX}

mlflow_logger = MLflowLogger(
        "Neural Talk Head - 2",
        'images-{}'.format(CELEB_IDX),
        params=mlflow_params,
)
load_dotenv(".env")

mlflow.log_artifacts(image_dir, "Synthesized_images")
mlflow.log_artifacts(descriptor_dir, "Descriptors")
mlflow.log_artifacts(landmark_dir, "Landmarks")
