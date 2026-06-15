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
from evaluate import PatchCoreEvaluator

LOGGER = logging.getLogger(__name__)

SAMPLER_MAP = {
    "identity":               lambda pct, dev: sampler.IdentitySampler(),
    "greedy_coreset":         lambda pct, dev: sampler.GreedyCoresetSampler(pct, dev),
    "approx_greedy_coreset":  lambda pct, dev: sampler.ApproximateGreedyCoresetSampler(pct, dev),
}


# ---------------------------------------------------------------------------
# DataPrefetcher  (similar to YOLO utils/dataloaders.py InfiniteDataLoader)
# ---------------------------------------------------------------------------

class DataPrefetcher:
    """Wrap a DataLoader and move batches to GPU in a non-blocking way.

    Uses ``.to(device, non_blocking=True)`` to move tensors to the GPU.
    ``pin_memory=True`` in the DataLoader enables efficient DMA transfers.
    ``persistent_workers=True`` avoids respawning subprocesses every iteration.

    IMPORTANT: We deliberately **do not** use a separate CUDA stream here.
    The combination of ``persistent_workers`` and multiple CUDA streams can
    cause deadlocks on some PyTorch versions (e.g. worker waits on stream A,
    main thread waits on stream B).
    """

    def __init__(self, loader: torch.utils.data.DataLoader,
                 device: torch.device):
        self.loader  = loader
        self.device  = device
        self.is_cuda = device.type == "cuda"
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)
        # Propagate the ``name`` attribute from the original loader if present
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
        """Move all tensors in a batch to the configured device (non-blocking)."""
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
    pc.add_argument("--patchstride", type=int, default=4,
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
    # --- Memory bank backend & segmentation training flags ---
    pc.add_argument("--memory_bank_backend", type=str, default="ram", choices=["ram", "disk"],
                    help="Backend penyimpanan fitur patch training: 'ram' (default, cepat, boros RAM) atau 'disk' (hemat RAM, lambat, cache di .cache/patchcore/)")
    pc.add_argument("--train_segmentation", action="store_true", default=False,
                    help="Jika diset, lakukan training & evaluasi segmentasi (pixel-level). Jika tidak, hanya image-level anomaly detection.")
    return p


# ---------------------------------------------------------------------------
# DataLoader builder  (DDP-aware + persistent_workers + prefetch_factor)
# ---------------------------------------------------------------------------

def _make_loader(dataset, batch_size: int, num_workers: int,
                 prefetch_factor: int, sampler_obj=None,
                 shuffle: bool = False) -> torch.utils.data.DataLoader:
    """Single factory for all DataLoaders with optimized settings.

    persistent_workers=True
        Keep worker subprocesses alive across iterations. Without this,
        PyTorch respawns all workers each time ``__iter__`` is called, which
        adds noticeable startup overhead on small per-class datasets like MVTec.

    prefetch_factor=N
        Each worker preloads N upcoming batches into CPU memory before they
        are requested.

    pin_memory=True
        Allocate batches in page-locked (pinned) memory. Transfers from pinned
        memory to GPU are much faster because DMA can access them without an
        additional copy.
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
    """Create DataLoaders for each sub-dataset and wrap them with
    :class:`DataPrefetcher` so CPU→GPU transfer overlaps with computation.
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
            # DistributedSampler: each rank only sees its own slice of the data
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
            # Manual multi-GPU: increase batch size according to the number of GPUs
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

        # ── Optional validation loader ───────────────────────────────────
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

        # ── Wrap all loaders with DataPrefetcher ─────────────────────────
        # DataPrefetcher moves batches to the GPU in a non-blocking way
        # using a combination of pinned memory (pin_memory=True) and
        # .to(device, non_blocking=True). This allows CPU→GPU transfer to
        # overlap with computation without needing a separate CUDA stream.
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
    """Build a list of :class:`PatchCore` instances.

    Multi-GPU strategy for PatchCore:

    DDP (``torchrun``):
        The backbone is wrapped with :class:`DistributedDataParallel`.
        Each process owns exactly one GPU.

    Manual multi-GPU (``--gpu 0 1`` without torchrun):
        We **do not** use ``torch.nn.DataParallel`` because the
        ForwardHook + exception-based early-exit pattern is incompatible
        with the way DataParallel replicates modules.

        Instead, :class:`common.MultiGPUFeatureExtractor` is used. It
        manually splits the batch across GPUs via a ``ThreadPoolExecutor``;
        each GPU has its own :class:`NetworkFeatureAggregator` and hooks.

        The list of ``gpu_ids`` is passed down into ``PatchCore.load()``
        and used there.
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
# DDP feature gathering
# ---------------------------------------------------------------------------

def gather_features_ddp(local_features: np.ndarray) -> np.ndarray:
    """Gather features from all ranks to rank 0 via ``all_gather``."""
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
    """DDP-aware equivalent of :meth:`PatchCore.fit`.

    1. Each rank extracts features from its own data slice (via
       :class:`DistributedSampler`).
    2. Features are gathered from all ranks.
    3. Rank 0 runs coreset sampling and builds the FAISS index.
    4. The FAISS index is broadcast to all ranks.
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
                # Images may already be on the GPU because of DataPrefetcher;
                # make sure they are float tensors and on the correct device.
                img = image.to(torch.float).to(pc.device)
                # ``_embed`` returns an np.ndarray of shape [N_patches, target_dim]
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
    # result_collect  = []

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
                # Sinkronisasi semua rank sebelum evaluasi & penyimpanan
                dist.barrier()

            if is_main_process() or not is_dist_active():
                model_dir = _save_patchcore_models(patchcore_list, run_save_path, dataset_name)

                evaluator = PatchCoreEvaluator(
                    model_dirs=[model_dir],
                    data_path=args.data_path,
                    results_path=run_save_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    resize=args.resize,
                    imagesize=args.imagesize,
                    faiss_on_gpu=args.faiss_on_gpu if hasattr(args, "faiss_on_gpu") else True,
                    faiss_num_workers=args.faiss_num_workers if hasattr(args, "faiss_num_workers") else 8,
                    device=device,
                    save_segmentation_images=args.segment if hasattr(args, "segment") else False,
                )
                result_collect = evaluator.evaluate()
                ml_logs.run_mlflow(args, run_save_path, result_collect)

    teardown_ddp()

def _save_patchcore_models(patchcore_list, run_save_path, dataset_name):
    save_path = os.path.join(run_save_path, "models", dataset_name)
    os.makedirs(save_path, exist_ok=True)
    n = len(patchcore_list)
    for i, pc in enumerate(patchcore_list):
        prepend = f"Ensemble-{i + 1}-{n}_" if n > 1 else ""
        pc.save_to_path(save_path, prepend)
    return save_path

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