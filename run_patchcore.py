import argparse
import contextlib
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from src.patchcore import patchcore, backbones, common, sampler, metrics, utils
from src.patchcore.datasets.mvtec import MVTecDataset
from src.helper import ml_logs

LOGGER = logging.getLogger(__name__)

SAMPLER_MAP = {
    "identity":               lambda pct, dev: sampler.IdentitySampler(),
    "greedy_coreset":         lambda pct, dev: sampler.GreedyCoresetSampler(pct, dev),
    "approx_greedy_coreset":  lambda pct, dev: sampler.ApproximateGreedyCoresetSampler(pct, dev),
}


# ---------------------------------------------------------------------------
# DataPrefetcher  (mirip YOLO utils/dataloaders.py InfiniteDataLoader)
# ---------------------------------------------------------------------------

class DataPrefetcher:
    """
    Membungkus DataLoader dengan transfer CPU→GPU non-blocking.

    Menggunakan .to(device, non_blocking=True) untuk memindahkan tensor ke GPU.
    pin_memory=True di DataLoader memastikan DMA transfer yang efisien.
    persistent_workers=True mencegah respawn subprocess tiap iterasi.

    PENTING: Tidak menggunakan CUDA stream terpisah karena kombinasi
    persistent_workers + multi-stream dapat menyebabkan deadlock pada
    beberapa versi PyTorch (worker menunggu di stream A, main thread di stream B).
    """

    def __init__(self, loader: torch.utils.data.DataLoader,
                 device: torch.device):
        self.loader  = loader
        self.device  = device
        self.is_cuda = device.type == "cuda"
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)
        # Propagasikan atribut name dari loader asli jika ada
        if hasattr(loader, "name"):
            self.name = loader.name

    def __iter__(self):
        for batch in self.loader:
            if self.is_cuda:
                yield self._to_device(batch)
            else:
                yield batch

    def __len__(self):
        return len(self.loader)

    def _to_device(self, batch):
        """Pindahkan semua tensor ke device dengan non_blocking=True."""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            moved = [
                v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for v in batch
            ]
            return type(batch)(moved)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch


def wrap_prefetcher(loader: torch.utils.data.DataLoader,
                    device: torch.device) -> DataPrefetcher:
    """Bungkus DataLoader dengan DataPrefetcher."""
    return DataPrefetcher(loader, device)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_active() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_active() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(local_rank: int) -> torch.device:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    LOGGER.info("DDP initialised: rank %d / %d", get_rank(), get_world_size())
    return torch.device(f"cuda:{local_rank}")


def teardown_ddp():
    if is_dist_active():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PatchCore anomaly detection pipeline — Multi-GPU + Prefetch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Global ---
    p.add_argument("--gpu", type=int, nargs="+", default=[0], metavar="GPU",
                   help="GPU id(s). Contoh: --gpu 0 1 2 3")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--log_project", type=str, default="project")
    p.add_argument("--local_rank",  type=int, default=-1,
                   help="Diisi otomatis oleh torchrun.")
    p.add_argument("--segment", action="store_true")

    # --- Dataset ---
    ds = p.add_argument_group("dataset")
    ds.add_argument("--data_path", "-d",       type=str,   required=True)
    ds.add_argument("--train_val_split", type=float, default=1.0)
    ds.add_argument("--batch_size", "-bs",      type=int,   default=16,
                    help="Batch size per GPU.")
    ds.add_argument("--num_workers", "-nw",     type=int,   default=8,
                    help="Jumlah subprocess worker per DataLoader.")
    ds.add_argument("--prefetch_factor", type=int,   default=2,
                    help="Berapa batch yang di-prefetch per worker (>=2). "
                         "Set 2 untuk keseimbangan memory vs kecepatan.")
    ds.add_argument("--resize",    type=int, default=256)
    ds.add_argument("--imagesize", type=int, default=224)

    # --- Sampler ---
    sm = p.add_argument_group("sampler")
    sm.add_argument("--sampler_name",        type=str,   default="approx_greedy_coreset",
                    choices=list(SAMPLER_MAP.keys()))
    sm.add_argument("--coreset_percentage", "-p", type=float, default=0.1)

    # --- PatchCore model ---
    pc = p.add_argument_group("patchcore")
    pc.add_argument("--backbone_names",          "-b",  nargs="+", type=str, default=['resnet50'])
    pc.add_argument("--layers_to_extract_from",  "-le", nargs="+", type=str, default=['layer1'])
    pc.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    pc.add_argument("--target_embed_dimension",   type=int, default=1024)
    pc.add_argument("--preprocessing", type=str, default="mean", choices=["mean", "conv"])
    pc.add_argument("--aggregation",   type=str, default="mean", choices=["mean", "mlp"])
    pc.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
    pc.add_argument("--patchsize",   type=int, default=3)
    pc.add_argument("--patchstride", type=int, default=1,
                    help="Stride untuk patch extraction. "
                         "Naikkan (mis. 2 atau 4) untuk kurangi jumlah patch "
                         "dan percepat coreset sampling secara signifikan. "
                         "stride=1 -> 224x224 patch/gambar, "
                         "stride=2 -> 112x112, stride=4 -> 56x56.")
    pc.add_argument("--patchscore", type=str,   default="max")
    pc.add_argument("--patchoverlap",       type=float, default=0.0)
    pc.add_argument("--patchsize_aggregate", "-pa", nargs="+", type=int, default=[])
    pc.add_argument("--faiss_on_gpu",      action="store_true")
    pc.add_argument("--faiss_num_workers", type=int, default=8)
    return p


# ---------------------------------------------------------------------------
# DataLoader builder  (DDP-aware + persistent_workers + prefetch_factor)
# ---------------------------------------------------------------------------

def _make_loader(dataset, batch_size: int, num_workers: int,
                 prefetch_factor: int, sampler_obj=None,
                 shuffle: bool = False) -> torch.utils.data.DataLoader:
    """
    Factory tunggal untuk semua DataLoader dengan pengaturan optimal:

    persistent_workers=True
        Worker subprocess tetap hidup antar iterasi dataset.
        Tanpa ini, PyTorch me-respawn semua worker tiap kali __iter__
        dipanggil ulang — overhead startup yang terasa pada dataset kecil
        seperti MVTec per-class.

    prefetch_factor=N
        Tiap worker men-queue N batch berikutnya di memori CPU sebelum
        diminta.

    pin_memory=True
        Alokasikan batch di page-locked (pinned) memory.
        Transfer dari pinned memory ke GPU jauh lebih cepat karena
        DMA dapat langsung mengakses tanpa copy tambahan.
    """
    use_pin_memory = torch.cuda.is_available() and num_workers > 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_obj,
        shuffle=(shuffle and sampler_obj is None),
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def get_dataloaders(args: argparse.Namespace, seed: int,
                    device: torch.device,
                    ddp_mode: bool = False) -> list[dict]:
    """
    Buat DataLoader untuk setiap sub-dataset, lalu bungkus dengan
    DataPrefetcher supaya transfer CPU→GPU overlap dengan komputasi.
    """
    dataloaders = []
    subdatasets = sorted([
        fname for fname in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, fname))
    ])

    for subdataset in subdatasets:
        train_ds = MVTecDataset(
            args.data_path, classname=subdataset,
            resize=args.resize, imagesize=args.imagesize,
            train_val_split=args.train_val_split, split="train",
        )
        test_ds = MVTecDataset(
            args.data_path, classname=subdataset,
            resize=args.resize, imagesize=args.imagesize,
            split="test",
        )

        # ── Training loader ──────────────────────────────────────────────
        if ddp_mode:
            # DistributedSampler: tiap rank hanya dapat slice datanya sendiri
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                seed=seed,
            )
            train_bs = args.batch_size          # per-GPU
        else:
            train_sampler = None
            # DataParallel: batch diperbesar sesuai jumlah GPU
            train_bs = args.batch_size * max(1, len(args.gpu))

        train_loader = _make_loader(
            train_ds, train_bs,
            args.num_workers, args.prefetch_factor,
            sampler_obj=train_sampler,
        )
        train_loader.name = "mvtc" + (f"_{subdataset}" if subdataset else "")

        # ── Testing loader ───────────────────────────────────────────────
        test_loader = _make_loader(
            test_ds, args.batch_size,
            args.num_workers, args.prefetch_factor,
        )

        # ── Validation loader (opsional) ─────────────────────────────────
        val_loader = None
        if args.train_val_split < 1:
            val_ds = MVTecDataset(
                args.data_path, classname=subdataset,
                resize=args.resize, imagesize=args.imagesize,
                train_val_split=args.train_val_split, split="val",
            )
            val_loader = _make_loader(
                val_ds, args.batch_size,
                args.num_workers, args.prefetch_factor,
            )

        # ── Bungkus semua loader dengan DataPrefetcher ───────────────────
        # DataPrefetcher menggunakan CUDA stream terpisah untuk memindahkan
        # batch ke GPU di background — CPU dan GPU bekerja bersamaan.
        train_pf = wrap_prefetcher(train_loader, device)
        test_pf  = wrap_prefetcher(test_loader,  device)
        val_pf   = wrap_prefetcher(val_loader,   device) if val_loader else None

        dataloaders.append({
            "training":   train_pf,
            "validation": val_pf,
            "testing":    test_pf,
        })

    return dataloaders


# ---------------------------------------------------------------------------
# Sampler / PatchCore builder
# ---------------------------------------------------------------------------

def get_sampler(args: argparse.Namespace, device: torch.device):
    return SAMPLER_MAP[args.sampler_name](args.coreset_percentage, device)


def build_layers_per_backbone(backbone_names, layers_to_extract_from):
    if len(backbone_names) <= 1:
        return [list(layers_to_extract_from)]
    per_backbone = [[] for _ in backbone_names]
    for layer in layers_to_extract_from:
        idx, rest = int(layer.split(".")[0]), ".".join(layer.split(".")[1:])
        per_backbone[idx].append(rest)
    return per_backbone


def get_patchcore_list(args, input_shape, feat_sampler,
                       device, gpu_ids, ddp_mode=False):
    """
    Buat daftar PatchCore instance.

    Strategi multi-GPU yang benar untuk PatchCore:

    DDP (torchrun):
        Backbone di-wrap DDP. Setiap proses punya 1 GPU.

    DataParallel (--gpu 0 1, tanpa torchrun):
        TIDAK pakai torch.nn.DataParallel — ForwardHook + exception tidak
        kompatibel dengan cara DP mereplikasi modul.
        Gantinya: MultiGPUFeatureExtractor di common.py yang secara manual
        split batch ke tiap GPU via ThreadPoolExecutor, tiap GPU punya
        NetworkFeatureAggregator + hook sendiri yang independen.
        gpu_ids diteruskan ke PatchCore.load() dan dipakai di sana.
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

        if ddp_mode:
            backbone = DDP(backbone, device_ids=[device.index],
                           output_device=device.index,
                           find_unused_parameters=True)
            LOGGER.info("Backbone '%s' -> DDP (rank %d).", backbone_name, get_rank())
            effective_gpu_ids = None
        else:
            if len(gpu_ids) > 1:
                LOGGER.info("Backbone '%s' -> MultiGPUFeatureExtractor %s.",
                            backbone_name, gpu_ids)
            effective_gpu_ids = gpu_ids

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
            patchstride=args.patchstride,
            featuresampler=feat_sampler,
            nn_method=nn_method,
            gpu_ids=effective_gpu_ids,
        )
        instances.append(instance)

    return instances


# ---------------------------------------------------------------------------
# Score aggregation helpers
# ---------------------------------------------------------------------------

def _normalize_and_mean(arrays: np.ndarray) -> np.ndarray:
    mins = arrays.min(axis=-1, keepdims=True)
    maxs = arrays.max(axis=-1, keepdims=True)
    return np.mean((arrays - mins) / (maxs - mins), axis=0)


def aggregate_scores(raw):
    return _normalize_and_mean(np.array(raw))


def aggregate_segmentations(raw):
    arr  = np.array(raw)
    mins = arr.reshape(len(arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    maxs = arr.reshape(len(arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    return np.mean((arr - mins) / (maxs - mins), axis=0)


# ---------------------------------------------------------------------------
# DDP feature gathering
# ---------------------------------------------------------------------------

def gather_features_ddp(local_features: np.ndarray) -> np.ndarray:
    """Kumpulkan fitur dari semua rank ke rank 0 via all_gather."""
    if not is_dist_active() or get_world_size() == 1:
        return local_features

    tensor = torch.from_numpy(local_features).cuda()
    size   = torch.tensor([tensor.shape[0]], device=tensor.device)

    all_sizes = [torch.zeros_like(size) for _ in range(get_world_size())]
    dist.all_gather(all_sizes, size)

    max_size = max(s.item() for s in all_sizes)
    padded   = torch.zeros(max_size, *tensor.shape[1:],
                           dtype=tensor.dtype, device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded)

    return np.concatenate([
        g[:s.item()].cpu().numpy()
        for g, s in zip(gathered, all_sizes)
    ], axis=0)


# ---------------------------------------------------------------------------
# DDP-aware fit
# ---------------------------------------------------------------------------

def _fit_ddp(pc: patchcore.PatchCore,
             training_loader) -> None:
    """
    Versi DDP dari pc.fit():
      1. Tiap rank ekstrak fitur dari slice data miliknya (via DistributedSampler).
      2. Fitur di-gather dari semua rank.
      3. Rank 0 jalankan coreset sampling + bangun FAISS index.
      4. FAISS index di-broadcast ke semua rank.
    """
    import tqdm

    pc.forward_modules.eval()

    local_features = []
    with tqdm.tqdm(training_loader,
                   desc=f"[rank {get_rank()}] Computing features...",
                   position=get_rank(), leave=False) as it:
        for image in it:
            if isinstance(image, dict):
                image = image["image"]
            with torch.no_grad():
                # Gambar mungkin sudah di GPU karena DataPrefetcher;
                # pastikan float dan di device yang benar.
                img = image.to(torch.float).to(pc.device)
                # _embed mengembalikan np.ndarray [N_patches, target_dim]
                local_features.append(pc._embed(img))

    local_features = np.concatenate(local_features, axis=0)
    all_features   = gather_features_ddp(local_features)

    if is_main_process() or not is_dist_active():
        sampled = pc.featuresampler.run(all_features)
        pc.anomaly_scorer.fit(detection_features=[sampled])

    if is_dist_active() and get_world_size() > 1:
        _broadcast_faiss_index(pc)


def _broadcast_faiss_index(pc: patchcore.PatchCore) -> None:
    """Broadcast FAISS index dari rank 0 ke semua rank."""
    import faiss

    tmp_path = f"/tmp/_patchcore_faiss_rank{get_rank()}.index"

    if get_rank() == 0:
        idx_cpu = (
            faiss.index_gpu_to_cpu(pc.anomaly_scorer.nn_method.search_index)
            if pc.anomaly_scorer.nn_method.on_gpu
            else pc.anomaly_scorer.nn_method.search_index
        )
        faiss.write_index(idx_cpu, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        data_tensor = torch.tensor(list(data), dtype=torch.uint8).cuda()
    else:
        data_tensor = torch.zeros(1, dtype=torch.uint8).cuda()

    size = torch.tensor([data_tensor.shape[0]], device="cuda")
    dist.broadcast(size, src=0)

    if get_rank() != 0:
        data_tensor = torch.zeros(size.item(), dtype=torch.uint8, device="cuda")

    dist.broadcast(data_tensor, src=0)

    if get_rank() != 0:
        with open(tmp_path, "wb") as f:
            f.write(bytes(data_tensor.cpu().numpy().tolist()))
        idx = faiss.read_index(tmp_path)
        if pc.anomaly_scorer.nn_method.on_gpu:
            idx = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), get_rank(), idx
            )
        pc.anomaly_scorer.nn_method.search_index = idx

    dist.barrier()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ── Mode & device ────────────────────────────────────────────────────
    ddp_mode = args.local_rank >= 0
    dp_mode  = (not ddp_mode) and len(args.gpu) > 1

    if ddp_mode:
        device = setup_ddp(args.local_rank)
    elif args.gpu:
        device = torch.device(f"cuda:{args.gpu[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_main_process():
        run_save_path = utils.create_storage_folder(args.log_project)
        mode_str = "DDP" if ddp_mode else ("DataParallel" if dp_mode else "Single-GPU/CPU")
        LOGGER.info("Save path : %s", run_save_path)
        LOGGER.info("Mode      : %s | Device: %s", mode_str, device)
        LOGGER.info("Workers   : %d × prefetch_factor=%d | DataPrefetcher: %s",
                    args.num_workers, args.prefetch_factor,
                    "ON (CUDA)" if device.type == "cuda" else "OFF (CPU pass-through)")
    else:
        run_save_path = None

    if is_dist_active():
        dist.barrier()
        obj = [run_save_path]
        dist.broadcast_object_list(obj, src=0)
        run_save_path = obj[0]

    device_context = (
        torch.cuda.device(device) if "cuda" in str(device)
        else contextlib.suppress()
    )

    # DataLoader sudah dibungkus DataPrefetcher di dalam get_dataloaders()
    all_dataloaders = get_dataloaders(args, args.seed, device, ddp_mode=ddp_mode)
    result_collect  = []

    for idx, dataloaders in enumerate(all_dataloaders):
        dataset_name = dataloaders["training"].name
        if is_main_process():
            LOGGER.info("Dataset [%s] (%d/%d)...",
                        dataset_name, idx + 1, len(all_dataloaders))
        utils.fix_seeds(args.seed, device)

        # Tiap epoch DistributedSampler perlu di-set agar urutan berbeda
        raw_train_loader = dataloaders["training"].loader
        if ddp_mode and hasattr(raw_train_loader.sampler, "set_epoch"):
            raw_train_loader.sampler.set_epoch(idx)

        with device_context:
            torch.cuda.empty_cache()
            imagesize    = dataloaders["training"].dataset.imagesize
            feat_sampler = get_sampler(args, device)
            patchcore_list = get_patchcore_list(
                args, imagesize, feat_sampler, device, args.gpu, ddp_mode=ddp_mode
            )

            if is_main_process() and len(patchcore_list) > 1:
                LOGGER.info("PatchCore Ensemble (N=%d).", len(patchcore_list))

            # ── Training ─────────────────────────────────────────────────
            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                raw_bb = pc._raw_backbone
                if getattr(raw_bb, "seed", None) is not None:
                    utils.fix_seeds(raw_bb.seed, device)
                if is_main_process():
                    LOGGER.info("Training model (%d/%d)", i + 1, len(patchcore_list))

                if ddp_mode:
                    _fit_ddp(pc, dataloaders["training"])
                else:
                    pc.fit(dataloaders["training"])

            if is_dist_active():
                dist.barrier()

            # ── Inference ────────────────────────────────────────────────
            torch.cuda.empty_cache()
            raw_scores, raw_segs = [], []
            labels_gt = masks_gt = None

            for i, pc in enumerate(patchcore_list):
                torch.cuda.empty_cache()
                if is_main_process():
                    LOGGER.info("Embedding test data (%d/%d)", i + 1, len(patchcore_list))
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

                if args.segment:
                    _save_segmentation_images(
                        args, dataloaders, run_save_path,
                        dataset_name, segmentations, scores,
                    )

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
                    "dataset_name":       dataset_name,
                    "instance_auroc":     auroc,
                    "full_pixel_auroc":   full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
                result_collect.append(result)
                for key, val in result.items():
                    if key != "dataset_name":
                        LOGGER.info("%s: %.3f", key, val)

                _save_patchcore_models(patchcore_list, run_save_path, dataset_name)

        if is_main_process():
            LOGGER.info("\n\n-----\n")

    if is_main_process() and result_collect:
        metric_names  = list(result_collect[-1].keys())[1:]
        dataset_names = [r["dataset_name"] for r in result_collect]
        scores_list   = [list(r.values())[1:] for r in result_collect]
        utils.compute_and_store_final_results(
            run_save_path, scores_list,
            column_names=metric_names, row_names=dataset_names,
        )
        ml_logs.run_mlflow(args, run_save_path, result_collect)

    teardown_ddp()

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_segmentation_images(args, dataloaders, run_save_path,
                               dataset_name, segmentations, scores):
    testing_ds  = dataloaders["testing"].dataset
    image_paths = [x[2] for x in testing_ds.data_to_iterate]
    mask_paths  = [x[3] for x in testing_ds.data_to_iterate]

    in_std  = np.array(testing_ds.transform_std).reshape(-1, 1, 1)
    in_mean = np.array(testing_ds.transform_mean).reshape(-1, 1, 1)

    def image_transform(image):
        # Denormalize: pixel = (normalized * std + mean) * 255
        img_tensor = testing_ds.transform_img(image).numpy()
        return np.clip((img_tensor * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][rank-%(process)d] %(levelname)s - %(message)s",
    )
    LOGGER.info("Command line arguments: %s", " ".join(sys.argv))
    args = build_parser().parse_args()

    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    run(args)