import math
import os
import pickle
import logging
from typing import List

import faiss
import numpy as np
import scipy.ndimage as ndimage
import threading
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

LOGGER = logging.getLogger(__name__)

class _FaissNN:
    def __init__(self, on_gpu: bool = False, num_workers: int = 4):
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu       = on_gpu
        self.search_index = None

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def _to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, faiss.GpuClonerOptions()
            )
        return index

    def run(self, n_nearest, query_features):
        return self.search_index.search(query_features, n_nearest)

    def load(self, filename: str):
        self.search_index = self._to_gpu(faiss.read_index(filename))

class _ConcatMerger:
    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)

    def merge(self, features: list):
        return np.concatenate([self._reduce(f) for f in features], axis=1)

class _NearestNeighbourScorer:
    def __init__(self, n_nearest_neighbours: int, nn_method: _FaissNN):
        self.feature_merger       = _ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method            = nn_method

    def predict(self, query_features: List[np.ndarray]):
        query        = self.feature_merger.merge(query_features)
        dists, _     = self.nn_method.run(self.n_nearest_neighbours, query)
        return np.mean(dists, axis=-1), dists, None

    def load(self, load_folder: str, prepend: str = ""):
        index_file = os.path.join(
            load_folder, prepend + "nnscorer_search_index.faiss"
        )
        self.nn_method.load(index_file)

class _MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class _Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList(
            [_MeanMapper(output_dim) for _ in input_dims]
        )

    def forward(self, features):
        return torch.stack(
            [mod(feat) for mod, feat in zip(self.preprocessing_modules, features)],
            dim=1,
        )


class _Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class _RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device      = device
        self.target_size = target_size
        self.smoothing   = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            s = patch_scores.to(self.device).unsqueeze(1)
            s = F.interpolate(
                s, size=self.target_size, mode="bilinear", align_corners=False
            )
            patch_scores = s.squeeze(1).cpu().numpy()
        return [
            ndimage.gaussian_filter(ps, sigma=self.smoothing)
            for ps in patch_scores
        ]

class _LastLayerReached(Exception):
    pass


class _ForwardHook:
    def __init__(self, hook_dict, layer_name, last_layer):
        self.hook_dict  = hook_dict
        self.layer_name = layer_name
        self.is_last    = (layer_name == last_layer)

    def __call__(self, module, input, output):
        outputs = getattr(self.hook_dict, "outputs", None)
        if outputs is None:
            return
        outputs[self.layer_name] = output
        if self.is_last:
            raise _LastLayerReached()


class _NetworkFeatureAggregator(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device):
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device   = device
        self._thread_local = threading.local()

        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        for layer in layers_to_extract_from:
            hook = _ForwardHook(self._thread_local, layer, layers_to_extract_from[-1])
            if "." in layer:
                block, idx = layer.split(".")
                net_layer  = self.backbone.__dict__["_modules"][block]
                net_layer  = (
                    net_layer[int(idx)] if idx.isnumeric()
                    else net_layer.__dict__["_modules"][idx]
                )
            else:
                net_layer = self.backbone.__dict__["_modules"][layer]

            handle = (
                net_layer[-1] if isinstance(net_layer, torch.nn.Sequential)
                else net_layer
            ).register_forward_hook(hook)
            self.backbone.hook_handles.append(handle)

        self.to(self.device)

    def forward(self, images):
        self._thread_local.outputs = {}
        with torch.no_grad():
            try:
                self.backbone(images)
            except _LastLayerReached:
                pass
        return self._thread_local.outputs

    def feature_dimensions(self, input_shape):
        out = self(torch.ones([1] + list(input_shape)).to(self.device))
        return [out[l].shape[1] for l in self.layers_to_extract_from]

class _PatchMaker:
    def __init__(self, patchsize, stride=1):
        self.patchsize = patchsize
        self.stride    = stride

    def patchify(self, features, return_spatial_info=False):
        padding  = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride,
            padding=padding, dilation=1,
        )
        unfolded   = unfolder(features)
        n_patches  = []
        for s in features.shape[-2:]:
            n = (s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            n_patches.append(int(n))
        unfolded = unfolded.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        return (unfolded, n_patches) if return_spatial_info else unfolded

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = isinstance(x, np.ndarray)
        if was_numpy:
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        return x.numpy() if was_numpy else x


def _to_spatial(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        B, N, D = x.shape
        patches = x[:, 1:, :]
        n       = patches.shape[1]
        grid    = int(math.isqrt(n))
        return patches.reshape(B, grid, grid, D).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Feature tensor harus 3D atau 4D, dapat: {x.shape}")

def _load_backbone(name: str) -> torch.nn.Module:
    import torchvision.models as models

    try:
        models.resnet50(weights="DEFAULT")
        wp = "weights='DEFAULT'"
    except TypeError:
        wp = "pretrained=True"

    cnn = {
        "alexnet":       f"models.alexnet({wp})",
        "resnet50":      f"models.resnet50({wp})",
        "resnet101":     f"models.resnet101({wp})",
        "resnext101":    f"models.resnext101_32x8d({wp})",
        "vgg11":         f"models.vgg11({wp})",
        "vgg19":         f"models.vgg19({wp})",
        "vgg19_bn":      f"models.vgg19_bn({wp})",
        "wideresnet50":  f"models.wide_resnet50_2({wp})",
        "wideresnet101": f"models.wide_resnet101_2({wp})",
    }
    if name in cnn:
        return eval(cnn[name])

    import timm
    timm_map = {
        "resnet200":          "resnet200",
        "resnest50":          "resnest50d_4s2x40d",
        "resnetv2_50_bit":    "resnetv2_50x3_bitm",
        "resnetv2_50_21k":    "resnetv2_50x3_bitm_in21k",
        "resnetv2_101_bit":   "resnetv2_101x3_bitm",
        "resnetv2_101_21k":   "resnetv2_101x3_bitm_in21k",
        "resnetv2_152_bit":   "resnetv2_152x4_bitm",
        "resnetv2_152_21k":   "resnetv2_152x4_bitm_in21k",
        "resnetv2_152_384":   "resnetv2_152x2_bit_teacher_384",
        "resnetv2_101":       "resnetv2_101",
        "mnasnet_100":        "mnasnet_100",
        "mnasnet_a1":         "mnasnet_a1",
        "mnasnet_b1":         "mnasnet_b1",
        "densenet121":        "densenet121",
        "densenet201":        "densenet201",
        "inception_v4":       "inception_v4",
        "vit_small":          "vit_small_patch16_224",
        "vit_base":           "vit_base_patch16_224",
        "vit_large":          "vit_large_patch16_224",
        "vit_r50":            "vit_large_r50_s32_224",
        "vit_deit_base":      "deit_base_patch16_224",
        "vit_deit_distilled": "deit_base_distilled_patch16_224",
        "vit_swin_base":      "swin_base_patch4_window7_224",
        "vit_swin_large":     "swin_large_patch4_window7_224",
        "efficientnet_b7":    "tf_efficientnet_b7",
        "efficientnet_b5":    "tf_efficientnet_b5",
        "efficientnet_b3":    "tf_efficientnet_b3",
        "efficientnet_b1":    "tf_efficientnet_b1",
        "efficientnetv2_m":   "tf_efficientnetv2_m",
        "efficientnetv2_l":   "tf_efficientnetv2_l",
        "efficientnet_b3a":   "efficientnet_b3a",
    }
    if name in timm_map:
        return timm.create_model(timm_map[name], pretrained=True)

    raise ValueError(
        f"Backbone '{name}' tidak dikenali. "
        f"Pilihan: {sorted(list(cnn) + list(timm_map))}"
    )

class _PatchCoreInference:
    """
    Inference engine PatchCore — instantiate via load_model(), bukan langsung.
    """

    def __init__(self, backbone, layers, device, input_shape,
                 pretrain_embed_dim, target_embed_dim,
                 patchsize, patchstride, n_nn,
                 faiss_index_path, prepend=""):

        self.device      = device
        self.patch_maker = _PatchMaker(patchsize, stride=patchstride)
        self._layers     = layers

        self._aggregator = _NetworkFeatureAggregator(backbone, layers, device)
        self._aggregator.eval()
        feat_dims = self._aggregator.feature_dimensions(input_shape)

        self._preprocessing = _Preprocessing(feat_dims, pretrain_embed_dim).to(device)
        self._preadapt      = _Aggregator(target_embed_dim).to(device)
        self._segmentor     = _RescaleSegmentor(device, target_size=input_shape[-2:])

        nn_method    = _FaissNN(on_gpu=False, num_workers=4)
        self._scorer = _NearestNeighbourScorer(n_nn, nn_method)
        self._scorer.load(faiss_index_path, prepend)

    def _embed(self, images: torch.Tensor):
        self._aggregator.eval()
        with torch.no_grad():
            feat_dict = self._aggregator(images)

        features    = [_to_spatial(feat_dict[l]) for l in self._layers]
        features    = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features     = [x[0] for x in features]
        ref          = patch_shapes[0]

        for i in range(1, len(features)):
            f, dims = features[i], patch_shapes[i]
            f = f.reshape(f.shape[0], dims[0], dims[1], *f.shape[2:])
            f = f.permute(0, -3, -2, -1, 1, 2)
            base = f.shape
            f = f.reshape(-1, *f.shape[-2:])
            f = F.interpolate(
                f.unsqueeze(1), size=(ref[0], ref[1]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            f = f.reshape(*base[:-2], ref[0], ref[1])
            f = f.permute(0, -2, -1, 1, 2, 3)
            features[i] = f.reshape(len(f), -1, *f.shape[-3:])

        features = [x.reshape(-1, *x.shape[-3:]).to(self.device) for x in features]
        features = self._preprocessing(features)
        features = self._preadapt(features)
        return features.detach().cpu().numpy(), patch_shapes

    def predict_tensor(self, image_tensor: torch.Tensor) -> dict:
        """
        Predict dari tensor [1, C, H, W].
        Returns {"score": float, "mask": np.ndarray [H, W]}.
        """
        images = image_tensor.to(torch.float).to(self.device)
        B      = images.shape[0]

        with torch.no_grad():
            features, patch_shapes = self._embed(images)
            patch_scores = self._scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=B)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            scales     = patch_shapes[0]
            seg        = self.patch_maker.unpatch_scores(patch_scores, batchsize=B)
            seg        = seg.reshape(B, scales[0], scales[1])
            masks      = self._segmentor.convert_to_segmentation(seg)

        return {
            "score": float(image_scores[0]),
            "mask":  np.array(masks[0]),
        }
    
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self,
                device: str = "cpu",
                prepend: str = "") -> _PatchCoreInference:
                
        params_file = os.path.join(self.model_path, prepend + "patchcore_params.pkl")

        with open(params_file, "rb") as f:
            params = pickle.load(f)

        LOGGER.info("Loading model: backbone=%s layers=%s",
                    params["backbone.name"], params["layers_to_extract_from"])

        backbone      = _load_backbone(params["backbone.name"])
        backbone.name = params["backbone.name"]

        _device  = torch.device(device)
        backbone = backbone.to(_device).eval()

        return _PatchCoreInference(
            backbone           = backbone,
            layers             = params["layers_to_extract_from"],
            device             = _device,
            input_shape        = params["input_shape"],
            pretrain_embed_dim = params["pretrain_embed_dimension"],
            target_embed_dim   = params["target_embed_dimension"],
            patchsize          = params["patchsize"],
            patchstride        = params["patchstride"],
            n_nn               = params["anomaly_scorer_num_nn"],
            faiss_index_path   = self.model_path,
            prepend            = prepend,
        )

    def predict_image(self,
                    image_path: str,
                    transform=None,
                    threshold: float = None) -> dict:
        model = self.load_model()
        tf = transform or _DEFAULT_TRANSFORM

        img = Image.open(image_path).convert("RGB")
        img.load()                              # eager decode, cegah lazy-load hang
        tensor = tf(img).unsqueeze(0)           # [1, C, H, W]

        result = model.predict_tensor(tensor)
        result["image_path"] = image_path
        result["is_anomaly"] = (
            bool(result["score"] > threshold) if threshold is not None else None
        )
        return result