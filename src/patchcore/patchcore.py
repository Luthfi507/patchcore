"""PatchCore and PatchCore detection methods — Multi-GPU aware."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel

from . import backbones
from . import common
from . import sampler


def _unwrap(module):
    """Kembalikan modul asli di balik DataParallel / DDP wrapper."""
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    return module


LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(False, 4),
        # Parameter baru untuk multi-GPU DP
        gpu_ids: list = None,
    ):
        self.backbone      = backbone
        self._raw_backbone = _unwrap(backbone)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape   = input_shape
        self.device        = device
        self.patch_maker   = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})

        # ── Feature Aggregator ────────────────────────────────────────────
        # Jika gpu_ids berisi >1 GPU, pakai MultiGPUFeatureExtractor yang
        # benar-benar membagi batch ke tiap GPU via ThreadPoolExecutor.
        # Jika single GPU / DDP, pakai NetworkFeatureAggregator biasa.
        self._gpu_ids = gpu_ids or []
        use_multi_gpu = len(self._gpu_ids) > 1 and not isinstance(
            backbone, DistributedDataParallel
        )

        if use_multi_gpu:
            # MultiGPUFeatureExtractor menangani hook + forward di semua GPU
            feature_aggregator = common.MultiGPUFeatureExtractor(
                backbone, layers_to_extract_from, self._gpu_ids
            )
            # Dapatkan dimensi fitur dari aggregator GPU-0
            feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
            # Simpan sebagai atribut biasa (bukan nn.Module) karena
            # MultiGPUFeatureExtractor tidak perlu ikut state_dict
            self._multi_gpu_aggregator = feature_aggregator
            self._use_multi_gpu = True
        else:
            feature_aggregator = common.NetworkFeatureAggregator(
                self._raw_backbone, layers_to_extract_from, device
            )
            feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
            self.forward_modules["feature_aggregator"] = feature_aggregator
            self._use_multi_gpu = False

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=device, target_size=input_shape[-2:]
        )
        self.featuresampler = featuresampler

    # ── Internal feature extractor (single call) ──────────────────────────

    def _get_raw_features(self, images: torch.Tensor) -> dict:
        """
        Jalankan backbone forward dan kembalikan dict {layer: tensor}.

        - Mode multi-GPU : MultiGPUFeatureExtractor (split batch antar GPU)
        - Mode single    : NetworkFeatureAggregator biasa
        """
        if self._use_multi_gpu:
            return self._multi_gpu_aggregator(images)
        else:
            return self.forward_modules["feature_aggregator"](images)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader) or hasattr(data, "loader"):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return np.concatenate(features, axis=0)
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images.

        Return shape (dengan detach=True):
          - provide_patch_shapes=False : np.ndarray [N_patches, target_dim]
          - provide_patch_shapes=True  : (np.ndarray [N_patches, target_dim], patch_shapes)
        """

        def _detach(features):
            if detach:
                # features = Tensor [N_patches, target_dim] → numpy 2D array
                return features.detach().cpu().numpy()
            return features

        if self._use_multi_gpu:
            self._multi_gpu_aggregator.eval()
        else:
            self.forward_modules["feature_aggregator"].eval()

        with torch.no_grad():
            # Panggil extractor yang tepat (multi atau single GPU)
            features_dict = self._get_raw_features(images)

        features = [features_dict[layer] for layer in self.layers_to_extract_from]
        features = [_to_spatial(x) for x in features]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features     = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features  = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # Pindahkan ke primary device sebelum masuk preprocessing
        features = [f.to(self.device) for f in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """
        Ekstrak fitur dari semua gambar training dan bangun FAISS index.

        Gambar bisa datang dari:
        - DataPrefetcher (sudah di GPU, non_blocking)
        - DataLoader biasa (di CPU)
        Keduanya ditangani dengan .to(torch.float).to(self.device).
        """
        if self._use_multi_gpu:
            self._multi_gpu_aggregator.eval()
        else:
            self.forward_modules.eval()

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    # Gambar mungkin sudah di GPU (dari DataPrefetcher) —
                    # .to() idempotent: tidak ada salinan jika sudah di device yang benar.
                    input_image = image.to(torch.float).to(self.device)
                    batch_features = self._embed(input_image)
                # _embed mengembalikan list of numpy arrays (satu per patch)
                # gunakan np.array() bukan extend agar axis tetap benar
                features.append(np.array(batch_features))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        # Terima DataLoader biasa ATAU DataPrefetcher wrapper
        if isinstance(data, torch.utils.data.DataLoader) or hasattr(data, "loader"):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        if self._use_multi_gpu:
            self._multi_gpu_aggregator.eval()
        else:
            self.forward_modules.eval()

        scores    = []
        masks     = []
        labels_gt = []
        masks_gt  = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # Tensor mungkin sudah di GPU (dari DataPrefetcher) — .cpu() sebelum .numpy()
                    labels_gt.extend(image["is_anomaly"].cpu().numpy().tolist())
                    masks_gt.extend(image["mask"].cpu().numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        images    = images.to(torch.float).to(self.device)
        if self._use_multi_gpu:
            self._multi_gpu_aggregator.eval()
        else:
            self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            # features: np.ndarray [N_patches, target_dim]
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales       = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks        = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(save_path, save_features_separately=False, prepend=prepend)
        raw_bb = self._raw_backbone
        patchcore_params = {
            "backbone.name":            raw_bb.name,
            "layers_to_extract_from":   self.layers_to_extract_from,
            "input_shape":              self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension":   self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize":                self.patch_maker.patchsize,
            "patchstride":              self.patch_maker.stride,
            "anomaly_scorer_num_nn":    self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, load_path, device, nn_method, prepend=""):
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = backbones.load(patchcore_params["backbone.name"])
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)
        self.anomaly_scorer.load(load_path, prepend)

def _to_spatial(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x 

    if x.ndim == 3:
        import math
        B, N, D = x.shape
        patch_tokens = x[:, 1:, :]          # [B, N-1, D]
        n_patches = patch_tokens.shape[1]
        grid = int(math.isqrt(n_patches))

        if grid * grid != n_patches:
            raise ValueError(
                f"ViT output memiliki {n_patches} patch tokens yang tidak "
                f"bisa di-reshape ke grid persegi (sqrt={grid:.2f}). "
                f"Pastikan imagesize adalah kelipatan patch size ViT (16)."
            )

        # [B, N-1, D] → [B, grid, grid, D] → [B, D, grid, grid]
        spatial = patch_tokens.reshape(B, grid, grid, D)
        spatial = spatial.permute(0, 3, 1, 2).contiguous()
        return spatial  # [B, D, grid, grid]

    raise ValueError(
        f"Feature tensor harus 3D [B,N,D] atau 4D [B,C,H,W], dapat shape: {x.shape}"
    )


class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride    = stride

    def patchify(self, features, return_spatial_info=False):
        padding  = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride,
            padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x