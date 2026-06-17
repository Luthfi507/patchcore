from loguru import logger
from dotenv import load_dotenv
import mlflow
import argparse
import os
from datetime import datetime
import pytz
import shutil
from time import time
from predict import Wrapper

load_dotenv()

file_path = os.path.abspath(__file__)
project_dir = os.path.abspath(
    os.path.join(os.path.dirname(file_path), '..', '..')
)
pred_path = os.path.join(project_dir, 'predict.py')
shutil.rmtree('mlflow_model', True)

run_name = str(datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Jakarta')).strftime("%d-%m-%y:%H-%M-%S-%f"))

def _mlflow_setup(args: argparse.Namespace):
    experiment = "patchcore experiment"
    mlflow.set_experiment(experiment)
    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"MLflow tracking URI: {tracking_uri}", )
    logger.info(f"MLflow experiment: {experiment}")

def _mlflow_log_params(args: argparse.Namespace):    
    params = {
        "backbone_names": " ".join(args.backbone_names),
        "layers_to_extract_from": " ".join(args.layers_to_extract_from),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "sampler_name": args.sampler_name,
        "coreset_percentage": args.coreset_percentage,
        "pretrain_embed_dimension": args.pretrain_embed_dimension,
        "target_embed_dimension": args.target_embed_dimension,
        "patchsize": args.patchsize,
        "patchstride": args.patchstride,
        "gpu_ids": " ".join(str(g) for g in args.gpu),
    }
    mlflow.log_params(params)

def _mlflow_log_metrics(dataset_name: str, result: dict):
    for key, val in result.items():
        if key == "dataset_name":
            continue
        import math
        if val is not None and not math.isnan(val):
            mlflow.log_metric(f"{dataset_name}/{key}", val)

def _mlflow_log_final(result_collect: list):
    if not result_collect:
        return
    
    import math
    metric_names = [k for k in result_collect[0] if k != "dataset_name"]
    for metric in metric_names:
        vals = [
            r[metric] for r in result_collect
            if r[metric] is not None and not math.isnan(r[metric])
        ]
        if vals:
            mlflow.log_metric(f"mean/{metric}", sum(vals) / len(vals))

def run_mlflow(args: argparse.Namespace, run_save_path: str, result_collect: list):
    start = time()
    _mlflow_setup(args)

    conda_env = {
        "name": "default",
        "channels": ["defaults"],
        "dependencies": [
            "python>=3.10.0",
            {
                "pip": [
                    "mlflow==2.3.2",
                    "setuptools==80.10.2",
                    "protobuf==4.25.8"
                ]
            }
        ]
    }
    
    with mlflow.start_run(run_name=run_name):
        logger.info("MLflow run started")

        _mlflow_log_params(args)

        for result in result_collect:
            dataset_name = result['dataset_name']
            _mlflow_log_metrics(dataset_name, result)
        
        _mlflow_log_final(result_collect)
        mlflow.log_artifact(os.path.join(run_save_path, 'results.csv'))

        model_dir = os.path.join(run_save_path, 'models', 'mvtc_screen')
        mlflow.pyfunc.log_model(
            artifact_path='pyfunc',
            python_model=Wrapper(),
            artifacts={'model_dir': model_dir},
            code_path=[pred_path],
            conda_env=conda_env
        )

        total_time = time() - start
        logger.success(f"MLflow run end in {total_time:.4f}")