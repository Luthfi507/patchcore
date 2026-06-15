import os
import gc
import logging
import numpy as np
import torch
import pickle

from src.patchcore import patchcore, common, metrics, utils
from src.patchcore.datasets.mvtec import MVTecDataset

LOGGER = logging.getLogger(__name__)

class PatchCoreEvaluator:
    def __init__(
        self,
        model_dirs,
        data_path,
        results_path="results",
        batch_size=1,
        num_workers=8,
        resize=256,
        imagesize=224,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        device=None,
        save_segmentation_images=False,
    ):
        self.model_dirs = model_dirs if isinstance(model_dirs, list) else [model_dirs]
        self.data_path = data_path
        self.results_path = results_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.imagesize = imagesize
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_segmentation_images = save_segmentation_images

    def _load_single_patchcore(self, path, prepend=""):
        nn_method = common.FaissNN(self.faiss_on_gpu, self.faiss_num_workers)
        instance = patchcore.PatchCore(self.device)
        params_path = os.path.join(path, f"{prepend}patchcore_params.pkl")
        with open(params_path, "rb") as f:
            patchcore_params = pickle.load(f)
        if "anomaly_scorer_num_nn" in patchcore_params:
            patchcore_params["anomaly_score_num_nn"] = patchcore_params.pop("anomaly_scorer_num_nn")
        filtered_params = {
            "backbone": patchcore.backbones.load(patchcore_params["backbone.name"]),
            "layers_to_extract_from": patchcore_params["layers_to_extract_from"],
            "device": self.device,
            "input_shape": patchcore_params["input_shape"],
            "pretrain_embed_dimension": patchcore_params["pretrain_embed_dimension"],
            "target_embed_dimension": patchcore_params["target_embed_dimension"],
            "patchsize": patchcore_params["patchsize"],
            "patchstride": patchcore_params["patchstride"],
            "anomaly_score_num_nn": patchcore_params["anomaly_score_num_nn"],
            "nn_method": nn_method,
        }
        filtered_params["backbone"].name = patchcore_params["backbone.name"]
        instance.load(**filtered_params)
        instance.anomaly_scorer.load(path, prepend)
        return instance

    def _get_subdatasets(self):
        return sorted([
            fname for fname in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, fname))
        ])

    def _dataloader_iter(self):
        for subdataset in self._get_subdatasets():
            test_ds = MVTecDataset(
                self.data_path,
                classname=subdataset,
                resize=self.resize,
                imagesize=self.imagesize,
                split="test",
            )
            loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            loader.name = "mvtc" + (f"_{subdataset}" if subdataset else "")
            yield {"testing": loader}

    def evaluate(self):
        os.makedirs(self.results_path, exist_ok=True)
        subdatasets = self._get_subdatasets()
        n_dataloaders = len(subdatasets)
        n_patchcores = len(self.model_dirs)
        if not (n_dataloaders == n_patchcores or n_patchcores == 1):
            raise ValueError("Please ensure that #PatchCores == #Datasets or #PatchCores == 1!")

        def patchcore_iter():
            for path in self.model_dirs:
                gc.collect()
                faiss_files = [f for f in os.listdir(path) if f.endswith(".faiss")]
                n = len(faiss_files)
                if n == 1:
                    instances = [self._load_single_patchcore(path)]
                else:
                    instances = [
                        self._load_single_patchcore(path, prepend=f"Ensemble-{i + 1}-{n}_")
                        for i in range(n)
                    ]
                yield instances

        pc_iter = patchcore_iter()
        dl_iter = self._dataloader_iter()

        result_collect = []
        current_patchcore_list = None

        for dl_count, dataloaders in enumerate(dl_iter):
            dataset_name = dataloaders["testing"].name
            LOGGER.info(
                "Evaluating dataset [%s] (%d/%d)...",
                dataset_name, dl_count + 1, n_dataloaders,
            )
            torch.cuda.empty_cache()

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

            scores = np.mean([np.asarray(s, dtype=np.float64) for s in raw_scores], axis=0)
            segmentations = np.mean([np.asarray(s, dtype=np.float64) for s in raw_segs], axis=0)
            anomaly_labels = labels_gt

            # --- (Optional) Save segmentation images ---
            if self.save_segmentation_images:
                self._save_segmentation_images(
                    self.results_path, dataloaders, segmentations, scores
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
            self.results_path,
            scores_list,
            column_names=metric_names,
            row_names=dataset_names,
        )
        return result_collect

    def _save_segmentation_images(self, results_path: str, dataloaders: dict,
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

if __name__ == "__main__":
    evaluator = PatchCoreEvaluator(
        model_dirs=["results/project/models/mvtc_toothbrush/"],
        data_path="mvtec_anomaly_detection/dataset/",
        results_path="results"
    )
    evaluator.evaluate()