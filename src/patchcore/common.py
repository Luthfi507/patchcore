import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel


def _unwrap_module(module):
    """Kembalikan modul asli di balik DataParallel / DDP wrapper."""
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    return module


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(self, n_nearest_neighbours, query_features, index_features=None):
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension), dimension, 512, 64, 8,
        )
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        pass

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(axis=-1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            self.preprocessing_modules.append(MeanMapper(output_dim))

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """
    Efficient extraction of network features.

    Perubahan vs versi asli:
    - Menerima backbone yang sudah di-unwrap (tanpa DP/DDP wrapper).
      Wrapping multi-GPU dilakukan DI LUAR oleh MultiGPUFeatureExtractor.
    - Hook dipasang ke backbone asli sehingga tidak ada konflik dengan DP.
    """

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        # Selalu pakai backbone asli (unwrap) untuk pemasangan hook
        self.backbone = _unwrap_module(backbone)
        self.device = device

        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = self.backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    network_layer = network_layer[int(extract_idx)]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = self.backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class MultiGPUFeatureExtractor:
    """
    Menjalankan NetworkFeatureAggregator di beberapa GPU secara paralel.

    Cara kerja (mirip YOLO manual batch-split):
      1. Batch dibagi rata ke N GPU.
      2. Tiap GPU menjalankan forward() di thread terpisah (concurrent.futures).
      3. Hasil dikumpulkan dan di-concat kembali di CPU.

    Kenapa tidak pakai DataParallel langsung?
    - ForwardHook menulis ke self.outputs (shared dict) — tidak thread-safe di DP.
    - LastLayerToExtractReachedException tidak bisa di-raise dari replika DP.
    - Solusi: satu NetworkFeatureAggregator per GPU, masing-masing punya
      hook dan outputs-nya sendiri, dijalankan paralel via ThreadPoolExecutor.
    """

    def __init__(self, backbone, layers_to_extract_from, gpu_ids: list):
        """
        Args:
            backbone   : backbone asli (belum di device manapun, atau sudah di gpu_ids[0]).
            layers_to_extract_from: list layer name.
            gpu_ids    : list GPU id, e.g. [0, 1].
        """
        self.gpu_ids = gpu_ids
        self.layers  = layers_to_extract_from
        self.n_gpus  = len(gpu_ids)

        # Buat satu salinan NetworkFeatureAggregator per GPU
        # Setiap aggregator punya backbone sendiri (deep copy) di GPU-nya
        raw_backbone = _unwrap_module(backbone)
        self.aggregators = []
        for i, gid in enumerate(gpu_ids):
            dev = torch.device(f"cuda:{gid}")
            if i == 0:
                bb = raw_backbone.to(dev)
            else:
                bb = copy.deepcopy(raw_backbone).to(dev)
            agg = NetworkFeatureAggregator(bb, layers_to_extract_from, dev)
            agg.eval()
            self.aggregators.append(agg)

        self.primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    def __call__(self, images: torch.Tensor) -> dict:
        """
        Forward images melalui semua GPU secara paralel.

        Args:
            images: tensor [B, C, H, W] di device manapun.
        Returns:
            dict {layer_name: tensor [B, C, H, W]} di primary_device.
        """
        import concurrent.futures

        B = images.shape[0]
        # Hitung ukuran chunk per GPU — sisa masuk ke GPU terakhir
        chunk_size = max(1, B // self.n_gpus)
        chunks     = []
        start      = 0
        for i in range(self.n_gpus):
            end = start + chunk_size if i < self.n_gpus - 1 else B
            if start < B:
                chunks.append(images[start:end])
            start = end

        actual_n = len(chunks)

        def _forward_one(agg, chunk):
            dev = agg.device
            with torch.no_grad():
                x = chunk.to(dev, non_blocking=True)
                return agg(x)          # returns dict {layer: tensor}

        # Jalankan paralel di thread pool (GIL tidak menghalangi CUDA ops)
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_n) as pool:
            futures = [
                pool.submit(_forward_one, self.aggregators[i], chunks[i])
                for i in range(actual_n)
            ]
            results = [f.result() for f in futures]

        # Gabungkan hasil dari semua GPU ke primary device
        merged = {}
        for layer in self.layers:
            parts = [
                res[layer].to(self.primary_device, non_blocking=True)
                for res in results
            ]
            merged[layer] = torch.cat(parts, dim=0)

        return merged

    def eval(self):
        for agg in self.aggregators:
            agg.eval()
        return self

    @property
    def device(self):
        return self.primary_device

    def feature_dimensions(self, input_shape):
        """Delegate ke aggregator pertama."""
        return self.aggregators[0].feature_dimensions(input_shape)


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method
        self.imagelevel_nn = lambda query: self.nn_method.run(n_nearest_neighbours, query)
        self.pixelwise_nn  = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

    def predict(self, query_features: List[np.ndarray]) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features  = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores  = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(self, save_folder, save_features_separately=False, prepend=""):
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(self._detection_file(save_folder, prepend), self.detection_features)

    def save_and_reset(self, save_folder):
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder, prepend=""):
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(self._detection_file(load_folder, prepend))