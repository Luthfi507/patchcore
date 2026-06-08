"""
PatchCore anomaly detection pipeline — Multi-GPU support.

Strategi multi-GPU (mirip YOLO):
  • DDP  (DistributedDataParallel) → dipakai kalau --local_rank diberikan
    melalui torchrun / torch.distributed.launch.
  • DP   (DataParallel)            → dipakai kalau lebih dari satu GPU
    diberikan via --gpu dan DDP tidak aktif.
  • CPU  → fallback kalau tidak ada GPU.

Cara pakai:
  # Single-GPU (lama, tetap bisa)
  python run_patchcore.py --gpu 0 --data_path /data ...

  # DataParallel (2 GPU, tanpa torchrun)
  python run_patchcore.py --gpu 0 1 --data_path /data ...

  # DistributedDataParallel (torchrun, recommended untuk banyak GPU)
  torchrun --nproc_per_node=2 run_patchcore.py --gpu 0 1 --data_path /data ...
"""

import argparse
import contextlib
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from src.patchcore import patchcore, backbones, common, sampler, metrics, utils
from src.patchcore.datasets.mvtec import MVTecDataset

LOGGER = logging.getLogger(__name__)

SAMPLER_MAP = {
    "identity": lambda pct, dev: sampler.IdentitySampler(),
    "greedy_coreset": lambda pct, dev: sampler.GreedyCoresetSampler(pct, dev),
    "approx_greedy_coreset": lambda pct, dev: sampler.ApproximateGreedyCoresetSampler(pct, dev),
}

# Helpers: distributed / device
def is_dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_active() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_active() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(local_rank: int) -> torch.device:
    """Inisialisasi proses DDP. Dipanggil dari tiap worker."""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    LOGGER.info("DDP initialised: rank %d / %d", get_rank(), get_world_size())
    return torch.device(f"cuda:{local_rank}")


def teardown_ddp():
    if is_dist_active():
        dist.destroy_process_group()

# Argument parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PatchCore anomaly detection pipeline — Multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Global ---
    p.add_argument("--gpu", type=int, nargs="+", default=[0], metavar="GPU",
                   help="GPU id(s) to use. E.g. --gpu 0 1 2 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_group", type=str, default="group")
    p.add_argument("--log_project", type=str, default="project")

    # --- DDP (diisi otomatis oleh torchrun) ---
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Rank lokal proses DDP. Diisi otomatis oleh torchrun.")

    # --- Dataset ---
    ds = p.add_argument_group("dataset")
    ds.add_argument("--data_path", type=str, required=True,
                    help="Root directory of the dataset")
    ds.add_argument("--train_val_split", type=float, default=1.0)
    ds.add_argument("--batch_size", type=int, default=2,
                    help="Batch size *per GPU*. Total batch = batch_size × num_gpus")
    ds.add_argument("--num_workers", type=int, default=8)
    ds.add_argument("--resize", type=int, default=256)
    ds.add_argument("--imagesize", type=int, default=224)

    # --- Sampler ---
    sm = p.add_argument_group("sampler")
    sm.add_argument("--sampler_name", type=str, default="approx_greedy_coreset",
                    choices=list(SAMPLER_MAP.keys()))
    sm.add_argument("--coreset_percentage", "-p", type=float, default=0.1)

    # --- PatchCore model ---
    pc = p.add_argument_group("patchcore")
    pc.add_argument("--backbone_names", "-b", nargs="+", type=str, default=[])
    pc.add_argument("--layers_to_extract_from", "-le", nargs="+", type=str, default=[])
    pc.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    pc.add_argument("--target_embed_dimension", type=int, default=1024)
    pc.add_argument("--preprocessing", type=str, default="mean",
                    choices=["mean", "conv"])
    pc.add_argument("--aggregation", type=str, default="mean",
                    choices=["mean", "mlp"])
    pc.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
    pc.add_argument("--patchsize", type=int, default=3)
    pc.add_argument("--patchscore", type=str, default="max")
    pc.add_argument("--patchoverlap", type=float, default=0.0)
    pc.add_argument("--patchsize_aggregate", "-pa", nargs="+", type=int, default=[])
    pc.add_argument("--faiss_on_gpu", action="store_true")
    pc.add_argument("--faiss_num_workers", type=int, default=8)

    return p

# DataLoader builder  (DDP-aware)
def get_dataloaders(args: argparse.Namespace, seed: int,
                    ddp_mode: bool = False) -> list[dict]:
    """
    Buat dataloader untuk setiap sub-dataset.

    Pada mode DDP, training loader menggunakan DistributedSampler supaya
    setiap GPU hanya memproses bagian datanya sendiri (mirip YOLO DDP).
    Testing loader tetap sequential karena agregasi skor dilakukan di proses
    utama setelah barrier.
    """
    common_kwargs = dict(
        resize=args.resize,
        imagesize=args.imagesize,
        seed=seed,
    )

    dataloaders = []
    subdatasets = sorted([
        fname for fname in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, fname))
    ])

    for subdataset in subdatasets:
        train_ds = MVTecDataset(
            args.data_path,
            classname=subdataset,
            train_val_split=args.train_val_split,
            split="train",
        )
        test_ds = MVTecDataset(
            args.data_path,
            classname=subdataset,
            split="test",
        )

        # ── Training loader ──────────────────────────────────────────────
        if ddp_mode:
            # DistributedSampler membagi data secara merata ke tiap GPU
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                seed=seed,
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=train_sampler,        # <-- menggantikan shuffle
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            # DataParallel / single GPU: loader biasa
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=args.batch_size * max(1, len(args.gpu)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        train_loader.name = "mvtc" + (f"_{subdataset}" if subdataset else "")

        # ── Testing loader ───────────────────────────────────────────────
        # Testing selalu dijalankan di proses utama (atau di semua proses
        # kalau DDP, tapi kita barier + gather di akhir).
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # ── Validation loader (opsional) ─────────────────────────────────
        val_loader = None
        if args.train_val_split < 1:
            val_ds = MVTecDataset(
                args.data_path,
                classname=subdataset,
                train_val_split=args.train_val_split,
                split="val",
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        dataloaders.append({
            "training": train_loader,
            "validation": val_loader,
            "testing": test_loader,
        })

    return dataloaders

# Sampler / PatchCore builder
def get_sampler(args: argparse.Namespace, device: torch.device):
    factory = SAMPLER_MAP[args.sampler_name]
    return factory(args.coreset_percentage, device)


def build_layers_per_backbone(backbone_names: list[str],
                               layers_to_extract_from: list[str]) -> list[list[str]]:
    """Split layer specs (prefixed dengan backbone index) ke per-backbone lists."""
    if len(backbone_names) <= 1:
        return [list(layers_to_extract_from)]

    per_backbone: list[list[str]] = [[] for _ in backbone_names]
    for layer in layers_to_extract_from:
        idx, rest = int(layer.split(".")[0]), ".".join(layer.split(".")[1:])
        per_backbone[idx].append(rest)
    return per_backbone


def get_patchcore_list(args: argparse.Namespace,
                       input_shape,
                       feat_sampler,
                       device: torch.device,
                       gpu_ids: list[int],
                       ddp_mode: bool = False) -> list:
    """
    Buat daftar PatchCore instance.

    Multi-GPU wrapping:
      • DDP  → backbone di-wrap DDP supaya gradien tersinkronisasi.
               (PatchCore sendiri tidak pakai backward, tapi DDP masih berguna
               untuk sinkronisasi BN dan forward yang efisien.)
      • DP   → backbone di-wrap DataParallel; forward otomatis di-split ke
               semua GPU yang tersedia.
    """
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
        backbone = backbone.to(device)

        # ── Multi-GPU wrapping (mirip YOLO model.py) ─────────────────────
        if ddp_mode:
            # DDP: setiap proses sudah punya satu GPU
            backbone = DDP(
                backbone,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,   # backbone bisa punya layer idle
            )
            LOGGER.info("Backbone '%s' di-wrap dengan DDP (rank %d).",
                        backbone_name, get_rank())
        elif len(gpu_ids) > 1:
            # DataParallel: satu proses, banyak GPU
            backbone = DataParallel(backbone, device_ids=gpu_ids)
            LOGGER.info("Backbone '%s' di-wrap dengan DataParallel %s.",
                        backbone_name, gpu_ids)

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
            featuresampler=feat_sampler,
            nn_method=nn_method,
        )
        instances.append(instance)

    return instances

# Score aggregation helpers
def _normalize_and_mean(arrays: np.ndarray) -> np.ndarray:
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

# DDP feature gathering
def gather_features_ddp(local_features: np.ndarray) -> np.ndarray:
    """
    Kumpulkan fitur dari semua rank DDP ke rank 0.

    Mirip cara YOLO mengumpulkan hasil prediksi antar proses.
    """
    if not is_dist_active() or get_world_size() == 1:
        return local_features

    tensor = torch.from_numpy(local_features).cuda()

    # Beritahu semua proses berapa banyak elemen yang dimiliki masing-masing
    size = torch.tensor([tensor.shape[0]], device=tensor.device)
    all_sizes = [torch.zeros_like(size) for _ in range(get_world_size())]
    dist.all_gather(all_sizes, size)

    max_size = max(s.item() for s in all_sizes)

    # Pad agar ukuran sama (all_gather butuh tensor seragam)
    padded = torch.zeros(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype, device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded)

    # Potong padding, gabungkan
    result = np.concatenate([
        g[:s.item()].cpu().numpy()
        for g, s in zip(gathered, all_sizes)
    ], axis=0)
    return result

# Main pipeline
def run(args: argparse.Namespace) -> None:
    # ── Tentukan mode dan device ─────────────────────────────────────────
    ddp_mode = args.local_rank >= 0           # torchrun mengisi local_rank
    dp_mode  = (not ddp_mode) and len(args.gpu) > 1

    if ddp_mode:
        device = setup_ddp(args.local_rank)
    elif args.gpu:
        device = torch.device(f"cuda:{args.gpu[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Hanya proses utama yang membuat folder & menulis log utama
    if is_main_process():
        run_save_path = utils.create_storage_folder(
            args.log_project, args.log_group
        )
        LOGGER.info("Save path: %s", run_save_path)
        LOGGER.info("Mode : %s",
                    "DDP" if ddp_mode else ("DataParallel" if dp_mode else "Single-GPU/CPU"))
        LOGGER.info("Device: %s", device)
        if not ddp_mode and len(args.gpu) > 1:
            LOGGER.info("GPU ids: %s", args.gpu)
    else:
        run_save_path = None   # worker rank tidak butuh path ini

    # Sinkronkan semua proses sebelum mulai
    if is_dist_active():
        dist.barrier()
    # Broadcast run_save_path dari rank 0 ke semua rank (pakai object list)
    if is_dist_active():
        obj = [run_save_path]
        dist.broadcast_object_list(obj, src=0)
        run_save_path = obj[0]

    device_context = (
        torch.cuda.device(device)
        if "cuda" in str(device)
        else contextlib.suppress()
    )

    all_dataloaders = get_dataloaders(args, args.seed, ddp_mode=ddp_mode)
    result_collect = []

    for idx, dataloaders in enumerate(all_dataloaders):
        dataset_name = dataloaders["training"].name
        if is_main_process():
            LOGGER.info(
                "Evaluating dataset [%s] (%d/%d)...",
                dataset_name, idx + 1, len(all_dataloaders),
            )
        utils.fix_seeds(args.seed, device)

        # Set epoch pada DistributedSampler supaya shuffle berbeda tiap epoch
        if ddp_mode and hasattr(dataloaders["training"].sampler, "set_epoch"):
            dataloaders["training"].sampler.set_epoch(idx)

        with device_context:
            torch.cuda.empty_cache()
            imagesize  = dataloaders["training"].dataset.imagesize
            feat_sampler = get_sampler(args, device)
            patchcore_list = get_patchcore_list(
                args, imagesize, feat_sampler, device,
                args.gpu, ddp_mode=ddp_mode,
            )

            if is_main_process() and len(patchcore_list) > 1:
                LOGGER.info("Menggunakan PatchCore Ensemble (N=%d).", len(patchcore_list))

            # ── Training ─────────────────────────────────────────────────
            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                if hasattr(pc.backbone, "module"):
                    # Ambil backbone asli di balik DDP/DP wrapper
                    seed_attr = pc.backbone.module.seed
                else:
                    seed_attr = pc.backbone.seed

                if seed_attr is not None:
                    utils.fix_seeds(seed_attr, device)

                if is_main_process():
                    LOGGER.info("Training model (%d/%d)", i + 1, len(patchcore_list))

                # fit() sudah memanggil _fill_memory_bank yang iterasi DataLoader.
                # Pada DDP, setiap rank mengolah bagian datanya sendiri berkat
                # DistributedSampler, lalu fitur di-gather di sini.
                if ddp_mode:
                    _fit_ddp(pc, dataloaders["training"])
                else:
                    pc.fit(dataloaders["training"])

            # ── Barrier setelah training selesai ─────────────────────────
            if is_dist_active():
                dist.barrier()

            # ── Inference (hanya rank 0 untuk menghindari duplikasi) ──────
            torch.cuda.empty_cache()
            raw_scores, raw_segs = [], []
            labels_gt = masks_gt = None

            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                if is_main_process():
                    LOGGER.info("Embedding test data (%d/%d)", i + 1, len(patchcore_list))
                # Inferensi: cukup satu proses (rank 0) yang jalan;
                # di DDP, proses lain idle selama ini (pola umum di YOLO val).
                if is_main_process() or not is_dist_active():
                    scores, segmentations, labels_gt, masks_gt = pc.predict(
                        dataloaders["testing"]
                    )
                    raw_scores.append(scores)
                    raw_segs.append(segmentations)

            if is_dist_active():
                dist.barrier()

            if is_main_process() or not is_dist_active():
                scores        = aggregate_scores(raw_scores)
                segmentations = aggregate_segmentations(raw_segs)

                anomaly_labels = [
                    x[1] != "good"
                    for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                _save_segmentation_images(
                    args, dataloaders, run_save_path,
                    dataset_name, segmentations, scores,
                )

                # ── Metrics ───────────────────────────────────────────────
                LOGGER.info("Menghitung evaluation metrics.")
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

        if is_main_process():
            LOGGER.info("\n\n-----\n")

    # ── Final CSV (hanya rank 0) ──────────────────────────────────────────
    if is_main_process() and result_collect:
        metric_names  = list(result_collect[-1].keys())[1:]
        dataset_names = [r["dataset_name"] for r in result_collect]
        scores_list   = [list(r.values())[1:] for r in result_collect]
        utils.compute_and_store_final_results(
            run_save_path,
            scores_list,
            column_names=metric_names,
            row_names=dataset_names,
        )

    teardown_ddp()

# DDP-aware fit helper
def _fit_ddp(pc: patchcore.PatchCore,
             training_loader: torch.utils.data.DataLoader) -> None:
    """
    Versi DDP dari pc.fit():
      1. Tiap rank mengekstrak fitur dari bagian datanya sendiri.
      2. Fitur dikumpulkan dari semua rank (all_gather).
      3. Hanya rank 0 yang menjalankan coreset sampling & membangun FAISS index.
      4. Index di-broadcast ke semua rank supaya inferensi bisa berjalan di mana saja.
    """
    import tqdm

    pc.forward_modules.eval()

    def _img_to_feat(img):
        with torch.no_grad():
            img = img.to(torch.float).to(pc.device)
            return pc._embed(img)

    local_features = []
    with tqdm.tqdm(training_loader,
                   desc=f"[rank {get_rank()}] Computing features...",
                   position=get_rank(), leave=False) as it:
        for image in it:
            if isinstance(image, dict):
                image = image["image"]
            local_features.append(_img_to_feat(image))

    local_features = np.concatenate(local_features, axis=0)

    # Kumpulkan fitur dari semua rank
    all_features = gather_features_ddp(local_features)

    # Hanya rank 0 yang membangun memory bank (coreset + FAISS)
    if is_main_process() or not is_dist_active():
        sampled = pc.featuresampler.run(all_features)
        pc.anomaly_scorer.fit(detection_features=[sampled])

    # Broadcast FAISS index dari rank 0 ke semua rank
    # (agar inferensi bisa dijalankan di semua GPU kalau diperlukan)
    if is_dist_active() and get_world_size() > 1:
        _broadcast_faiss_index(pc)


def _broadcast_faiss_index(pc: patchcore.PatchCore) -> None:
    """
    Broadcast FAISS index dari rank 0 ke semua rank menggunakan pickle + tensor.
    """
    import io
    import faiss
    import pickle

    if get_rank() == 0:
        # Serialisasi index ke bytes
        buf = io.BytesIO()
        idx_cpu = faiss.index_gpu_to_cpu(pc.anomaly_scorer.nn_method.search_index) \
            if pc.anomaly_scorer.nn_method.on_gpu \
            else pc.anomaly_scorer.nn_method.search_index
        faiss.write_index(idx_cpu, "/tmp/_patchcore_faiss_tmp.index")
        with open("/tmp/_patchcore_faiss_tmp.index", "rb") as f:
            data = f.read()
        data_tensor = torch.tensor(list(data), dtype=torch.uint8).cuda()
    else:
        data_tensor = torch.zeros(1, dtype=torch.uint8).cuda()

    # Broadcast ukuran dulu
    size = torch.tensor([data_tensor.shape[0]], device="cuda")
    dist.broadcast(size, src=0)

    if get_rank() != 0:
        data_tensor = torch.zeros(size.item(), dtype=torch.uint8, device="cuda")

    dist.broadcast(data_tensor, src=0)

    # Rekonstruksi index di worker ranks
    if get_rank() != 0:
        raw = bytes(data_tensor.cpu().numpy().tolist())
        with open("/tmp/_patchcore_faiss_tmp.index", "wb") as f:
            f.write(raw)
        idx = faiss.read_index("/tmp/_patchcore_faiss_tmp.index")
        if pc.anomaly_scorer.nn_method.on_gpu:
            idx = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), get_rank(), idx
            )
        pc.anomaly_scorer.nn_method.search_index = idx

    dist.barrier()

# Save helpers
def _save_segmentation_images(args, dataloaders, run_save_path,
                               dataset_name, segmentations, scores):
    testing_ds  = dataloaders["testing"].dataset
    image_paths = [x[2] for x in testing_ds.data_to_iterate]
    mask_paths  = [x[3] for x in testing_ds.data_to_iterate]

    in_std  = np.array(testing_ds.transform_std).reshape(-1, 1, 1)
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

# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][rank-%(process)d] %(levelname)s - %(message)s",
    )
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    args = build_parser().parse_args()

    # Validasi: local_rank juga bisa datang dari env var (torchrun menyuntikkannya)
    if args.local_rank == -1:
        env_rank = os.environ.get("LOCAL_RANK", "-1")
        args.local_rank = int(env_rank)

    run(args)