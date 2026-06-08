import argparse
import contextlib
import logging
import os
import sys

import numpy as np
import torch

from src.patchcore import patchcore, backbones, common, sampler, metrics, utils
from src.patchcore.datasets.mvtec import MVTecDataset

LOGGER = logging.getLogger(__name__)

SAMPLER_MAP = {
    "identity": lambda pct, dev: sampler.IdentitySampler(),
    "greedy_coreset": lambda pct, dev: sampler.GreedyCoresetSampler(pct, dev),
    "approx_greedy_coreset": lambda pct, dev: sampler.ApproximateGreedyCoresetSampler(pct, dev),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PatchCore anomaly detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Global ---
    p.add_argument("--gpu", type=int, nargs="+", default=[0], metavar="GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_group", type=str, default="group")
    p.add_argument("--log_project", type=str, default="project")

    # --- Dataset ---
    ds = p.add_argument_group("dataset")
    ds.add_argument("--data_path", type=str, required=True, help="Root directory of the dataset")
    ds.add_argument("--train_val_split", type=float, default=1.0)
    ds.add_argument("--batch_size", type=int, default=2)
    ds.add_argument("--num_workers", type=int, default=8)
    ds.add_argument("--resize", type=int, default=256)
    ds.add_argument("--imagesize", type=int, default=224)

    # --- Sampler ---
    sm = p.add_argument_group("sampler")
    sm.add_argument("--sampler_name", type=str, default="approx_greedy_coreset", choices=list(SAMPLER_MAP.keys()))
    sm.add_argument("--coreset_percentage", "-p", type=float, default=0.1)

    # --- PatchCore model ---
    pc = p.add_argument_group("patchcore")
    pc.add_argument("--backbone_names", "-b", nargs="+", type=str, default=[])
    pc.add_argument("--layers_to_extract_from", "-le", nargs="+", type=str, default=[])
    pc.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    pc.add_argument("--target_embed_dimension", type=int, default=1024)
    pc.add_argument("--preprocessing", type=str, default="mean", choices=["mean", "conv"])
    pc.add_argument("--aggregation", type=str, default="mean", choices=["mean", "mlp"])
    pc.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
    pc.add_argument("--patchsize", type=int, default=3)
    pc.add_argument("--patchscore", type=str, default="max")
    pc.add_argument("--patchoverlap", type=float, default=0.0)
    pc.add_argument("--patchsize_aggregate", "-pa", nargs="+", type=int, default=[])
    pc.add_argument("--faiss_on_gpu", action="store_true")
    pc.add_argument("--faiss_num_workers", type=int, default=8)

    return p


def get_dataloaders(args: argparse.Namespace, seed: int) -> list[dict]:
    common_kwargs = dict(
        resize=args.resize,
        imagesize=args.imagesize,
        seed=seed,
    )

    dataloaders = []
    subdatasets = []

    for fname in os.listdir(args.data_path):
        project_path = os.path.join(args.data_path, fname)
        if os.path.isdir(project_path):
            subdatasets.append(fname)

    for subdataset in subdatasets:
        train_ds = MVTecDataset(
            args.data_path,
            classname=subdataset,
            train_val_split=args.train_val_split,
            split="train"
        )
        test_ds = MVTecDataset(
            args.data_path,
            classname=subdataset,
            split="test"
        )

        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
        test_loader = torch.utils.data.DataLoader(test_ds, **loader_kwargs)

        name = "mvtc" + (f"_{subdataset}" if subdataset else "")
        train_loader.name = name

        val_loader = None
        if args.train_val_split < 1:
            val_ds = MVTecDataset(
                args.data_path,
                classname=subdataset,
                train_val_split=args.train_val_split,
                split="val",
            )
            val_loader = torch.utils.data.DataLoader(val_ds, **loader_kwargs)

        dataloaders.append(
            {"training": train_loader, "validation": val_loader, "testing": test_loader}
        )
    return dataloaders


def get_sampler(args: argparse.Namespace, device: torch.device):
    factory = SAMPLER_MAP[args.sampler_name]
    return factory(args.coreset_percentage, device)


def build_layers_per_backbone(backbone_names: list[str],
                               layers_to_extract_from: list[str]) -> list[list[str]]:
    """Split layer specs (prefixed with backbone index) into per-backbone lists."""
    if len(backbone_names) <= 1:
        return [list(layers_to_extract_from)]

    per_backbone: list[list[str]] = [[] for _ in backbone_names]
    for layer in layers_to_extract_from:
        idx, rest = int(layer.split(".")[0]), ".".join(layer.split(".")[1:])
        per_backbone[idx].append(rest)
    return per_backbone


def get_patchcore_list(args: argparse.Namespace,
                       input_shape,
                       sampler,
                       device: torch.device) -> list:
    layers_per_backbone = build_layers_per_backbone(
        args.backbone_names, args.layers_to_extract_from
    )
    instances = []
    for backbone_name, layers in zip(args.backbone_names, layers_per_backbone):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = (
                backbone_name.split(".seed-")[0],
                int(backbone_name.split("-")[-1]),
            )

        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers)

        instance = patchcore.PatchCore(device)
        instance.load(
            backbone=backbone,
            layers_to_extract_from=layers,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            featuresampler=sampler,
            # anomaly_scorer_num_nn=args.anomaly_scorer_num_nn,
            nn_method=nn_method,
        )
        instances.append(instance)
    return instances

# Score aggregation helpers
def _normalize_and_mean(arrays: np.ndarray) -> np.ndarray:
    """Min-max normalize each row, then average across rows."""
    mins = arrays.min(axis=-1, keepdims=True)
    maxs = arrays.max(axis=-1, keepdims=True)
    return np.mean((arrays - mins) / (maxs - mins), axis=0)


def aggregate_scores(raw: list[np.ndarray]) -> np.ndarray:
    return _normalize_and_mean(np.array(raw))


def aggregate_segmentations(raw: list[np.ndarray]) -> np.ndarray:
    arr = np.array(raw)
    mins = arr.reshape(len(arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    maxs = arr.reshape(len(arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    return np.mean((arr - mins) / (maxs - mins), axis=0)

# Main pipeline
def run(args: argparse.Namespace) -> None:
    run_save_path = utils.create_storage_folder(
        args.log_project, args.log_group
    )

    device = utils.set_torch_device(args.gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    all_dataloaders = get_dataloaders(args, args.seed)
    result_collect = []

    for idx, dataloaders in enumerate(all_dataloaders):
        dataset_name = dataloaders["training"].name
        LOGGER.info(
            "Evaluating dataset [%s] (%d/%d)...",
            dataset_name, idx + 1, len(all_dataloaders),
        )
        utils.fix_seeds(args.seed, device)

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = get_sampler(args, device)
            patchcore_list = get_patchcore_list(args, imagesize, sampler, device)

            if len(patchcore_list) > 1:
                LOGGER.info("Utilizing PatchCore Ensemble (N=%d).", len(patchcore_list))

            # --- Training ---
            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                if pc.backbone.seed is not None:
                    utils.fix_seeds(pc.backbone.seed, device)
                LOGGER.info("Training models (%d/%d)", i + 1, len(patchcore_list))
                pc.fit(dataloaders["training"])

            # --- Inference ---
            torch.cuda.empty_cache()
            raw_scores, raw_segs = [], []
            labels_gt = masks_gt = None

            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                LOGGER.info("Embedding test data (%d/%d)", i + 1, len(patchcore_list))
                scores, segmentations, labels_gt, masks_gt = pc.predict(
                    dataloaders["testing"]
                )
                raw_scores.append(scores)
                raw_segs.append(segmentations)

            scores = aggregate_scores(raw_scores)
            segmentations = aggregate_segmentations(raw_segs)

            anomaly_labels = [
                x[1] != "good"
                for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # --- Save segmentation images ---
            _save_segmentation_images(
                args, dataloaders, run_save_path,
                dataset_name, segmentations, scores,
            )

            # --- Metrics ---
            LOGGER.info("Computing evaluation metrics.")
            auroc = metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            full_pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )["auroc"]

            anomaly_idxs = [i for i, m in enumerate(masks_gt) if np.sum(m) > 0]
            anomaly_pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in anomaly_idxs],
                [masks_gt[i] for i in anomaly_idxs],
            )["auroc"]

            result = {
                "dataset_name": dataset_name,
                "instance_auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,
            }
            result_collect.append(result)

            for key, val in result.items():
                if key != "dataset_name":
                    LOGGER.info("%s: %.3f", key, val)

            _save_patchcore_models(patchcore_list, run_save_path, dataset_name)

        LOGGER.info("\n\n-----\n")

    # --- Final CSV ---
    metric_names = list(result_collect[-1].keys())[1:]
    dataset_names = [r["dataset_name"] for r in result_collect]
    scores_list = [list(r.values())[1:] for r in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        scores_list,
        column_names=metric_names,
        row_names=dataset_names,
    )

def _save_segmentation_images(args, dataloaders, run_save_path, dataset_name, segmentations, scores):
    testing_ds = dataloaders["testing"].dataset
    image_paths = [x[2] for x in testing_ds.data_to_iterate]
    mask_paths = [x[3] for x in testing_ds.data_to_iterate]

    in_std = np.array(testing_ds.transform_std).reshape(-1, 1, 1)
    in_mean = np.array(testing_ds.transform_mean).reshape(-1, 1, 1)

    def image_transform(image):
        return np.clip(
            (testing_ds.transform_img(image).numpy() * in_std + in_mean) * 255,
            0, 255,
        ).astype(np.uint8)

    def mask_transform(mask):
        return testing_ds.transform_mask(mask).numpy()

    save_path = os.path.join(run_save_path, "segmentation_images", dataset_name)
    os.makedirs(save_path, exist_ok=True)
    utils.plot_segmentation_images(
        save_path, image_paths, segmentations, scores, mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )


def _save_patchcore_models(patchcore_list, run_save_path, dataset_name):
    save_path = os.path.join(run_save_path, "models", dataset_name)
    os.makedirs(save_path, exist_ok=True)
    n = len(patchcore_list)
    for i, pc in enumerate(patchcore_list):
        prepend = f"Ensemble-{i + 1}-{n}_" if n > 1 else ""
        pc.save_to_path(save_path, prepend)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    args = build_parser().parse_args()
    run(args)