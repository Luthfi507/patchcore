import logging
from dotenv import load_dotenv
import mlflow
import argparse

load_dotenv()
LOGGER = logging.getLogger(__name__)

def _mlflow_setup(args: argparse.Namespace):
    experiment = "patchcore experiment"
    mlflow.set_experiment(experiment)
    LOGGER.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    LOGGER.info("MLflow experiment: %s", experiment)

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
    _mlflow_setup(args)

    if not getattr(args, "no_mlflow", False):
        return
    
    with mlflow.start_run():
        LOGGER.info("MLflow run started: %s", mlflow.active_run().info.run_id)

        _mlflow_log_params(args)

        for result in result_collect:
            dataset_name = result['dataset_name']
            _mlflow_log_metrics(dataset_name, result)
        
        _mlflow_log_final(result_collect)

        mlflow.log_artifacts(run_save_path)
        LOGGER.info("MLflow run end.")