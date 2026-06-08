import argparse
import contextlib
import gc
import logging
import os
import sys

import numpy as np
import torch

from src.patchcore import patchcore, common, metrics, utils
from src.patchcore.datasets.mvtec import MVTecDataset

LOGGER = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load pre-trained PatchCore models and evaluate on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Global ---
    p.add_argument("results_path", type=str, help="Directory to store evaluation results")
    p.add_argument("--gpu", type=int, nargs="+", default=[0], metavar="GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_segmentation_images", action="store_true")

    # --- PatchCore loader ---
    pc = p.add_argument_group("patchcore_loader")
    pc.add_argument("--patch_core_paths", "-p", nargs="+", type=str, required=True, help="Paths to saved PatchCore model directories")
    pc.add_argument("--faiss_on_gpu", action="store_true")
    pc.add_argument("--faiss_num_workers", type=int, default=8)

    # --- Dataset ---
    ds = p.add_argument_group("dataset")
    ds.add_argument("--data_path", type=str, required=True, help="Root directory of the dataset")
    ds.add_argument("--batch_size", type=int, default=1)
    ds.add_argument("--num_workers", type=int, default=8)
    ds.add_argument("--resize", type=int, default=256)
    ds.add_argument("--imagesize", type=int, default=224)
    ds.add_argument("--augment", action="store_true")

    return p

# Validation
def validate_counts(n_dataloaders: int, n_patchcores: int) -> None:
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

# PatchCore loader
def _load_single_patchcore(path: str, device: torch.device, faiss_on_gpu: bool, faiss_num_workers: int, prepend: str = "") -> "patchcore.patchcore.PatchCore":
    nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)
    instance = patchcore.PatchCore(device)
    instance.load_from_path(
        load_path=path, device=device, nn_method=nn_method, prepend=prepend
    )
    return instance


def patchcore_iter(patch_core_paths: list[str], device: torch.device, faiss_on_gpu: bool, faiss_num_workers: int):
    """Yield one list of PatchCore instances per path entry."""
    for path in patch_core_paths:
        gc.collect()
        faiss_files = [f for f in os.listdir(path) if f.endswith(".faiss")]
        n = len(faiss_files)

        if n == 1:
            instances = [
                _load_single_patchcore(path, device, faiss_on_gpu, faiss_num_workers)
            ]
        else:
            instances = [
                _load_single_patchcore(
                    path, device, faiss_on_gpu, faiss_num_workers,
                    prepend=f"Ensemble-{i + 1}-{n}_",
                )
                for i in range(n)
            ]

        yield instances

# Dataloader factory
def dataloader_iter(args: argparse.Namespace, seed: int):
    """Yield one dataloader dict (test only) per subdataset."""
    subdatasets = []

    for fname in os.listdir(args.data_path):
        project_path = os.path.join(args.data_path, fname)
        if os.path.isdir(project_path):
            subdatasets.append(fname)

    for subdataset in args.subdatasets:
        test_ds = MVTecDataset(
            args.data_path,
            classname=subdataset,
            resize=args.resize,
            imagesize=args.imagesize,
            split="test",
            seed=seed,
        )
        loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        loader.name = "mvtc" + (f"_{subdataset}" if subdataset else "")
        yield {"testing": loader}

# Score aggregation helpers
def aggregate_scores(raw: list[np.ndarray]) -> np.ndarray:
    arr = np.array(raw)
    mins = arr.min(axis=-1, keepdims=True)
    maxs = arr.max(axis=-1, keepdims=True)
    return np.mean((arr - mins) / (maxs - mins), axis=0)


def aggregate_segmentations(raw: list[np.ndarray]) -> np.ndarray:
    arr = np.array(raw)
    mins = arr.reshape(len(arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    maxs = arr.reshape(len(arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    return np.mean((arr - mins) / (maxs - mins), axis=0)

# Side-effect helpers
def _save_segmentation_images(results_path: str, dataloaders: dict,
                               segmentations: np.ndarray, scores: np.ndarray) -> None:
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

    utils.plot_segmentation_images(
        results_path, image_paths, segmentations, scores, mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

# Main pipeline
def run(args: argparse.Namespace) -> None:
    os.makedirs(args.results_path, exist_ok=True)

    device = utils.set_torch_device(args.gpu)
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    n_dataloaders = len(args.subdatasets)
    n_patchcores = len(args.patch_core_paths)
    validate_counts(n_dataloaders, n_patchcores)

    pc_iter = patchcore_iter(
        args.patch_core_paths, device, args.faiss_on_gpu, args.faiss_num_workers
    )
    dl_iter = dataloader_iter(args, args.seed)

    result_collect = []
    current_patchcore_list = None

    for dl_count, dataloaders in enumerate(dl_iter):
        dataset_name = dataloaders["testing"].name
        LOGGER.info(
            "Evaluating dataset [%s] (%d/%d)...",
            dataset_name, dl_count + 1, n_dataloaders,
        )
        utils.fix_seeds(args.seed, device)

        with device_context:
            torch.cuda.empty_cache()

            # Advance patchcore iterator only when needed
            if dl_count < n_patchcores:
                current_patchcore_list = next(pc_iter)

            # --- Inference ---
            raw_scores, raw_segs = [], []
            labels_gt = masks_gt = None

            for i, pc in enumerate(current_patchcore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data (%d/%d)", i + 1, len(current_patchcore_list)
                )
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

            # --- (Optional) Save segmentation images ---
            if args.save_segmentation_images:
                _save_segmentation_images(
                    args.results_path, dataloaders, segmentations, scores
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

            del current_patchcore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    metric_names = list(result_collect[-1].keys())[1:]
    dataset_names = [r["dataset_name"] for r in result_collect]
    scores_list = [list(r.values())[1:] for r in result_collect]
    utils.compute_and_store_final_results(
        args.results_path,
        scores_list,
        column_names=metric_names,
        row_names=dataset_names,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    args = build_parser().parse_args()
    run(args)