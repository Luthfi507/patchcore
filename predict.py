import os
import sys
import argparse
import pickle
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from src.patchcore import patchcore, backbones, common, sampler, utils


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


# LOADER — load PatchCore object langsung dari repo
def load_patchcore(model_dir: str, device: torch.device) -> patchcore.PatchCore:
    params_path = os.path.join(model_dir, "patchcore_params.pkl")
    faiss_path  = None  # akan dicari otomatis


    # Baca params dulu untuk info
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    print("\n[MODEL] Parameter tersimpan:")
    for k, v in params.items():
        print(f"        {k}: {v}")

    # ── inisialisasi PatchCore object ──────────────────────────────────────────
    backbone_name = params.get("backbone.name", params.get("backbone_name", "wideresnet50"))
    layers        = list(params.get("layers_to_extract_from", ["layer2", "layer3"]))

    backbone = backbones.load(backbone_name)
    backbone.name = backbone_name

    loaded_patchcore = patchcore.PatchCore(device)
    loaded_patchcore.load(
        backbone                 = backbone,
        layers_to_extract_from   = layers,
        device                   = device,
        input_shape              = params.get("input_shape", (3, 224, 224)),
        pretrain_embed_dimension = params.get("pretrain_embed_dimension", 1024),
        target_embed_dimension   = params.get("target_embed_dimension", 1024),
        patchsize                = params.get("patchsize", 3),
        patchstride              = params.get("patchstride", 1),
        featuresampler           = sampler.IdentitySampler(),
        nn_method                = common.FaissNN(False, 8),
    )

    faiss_files = [f for f in os.listdir(model_dir) if f.endswith(".faiss")]
    if not faiss_files:
        raise FileNotFoundError(f"Tidak ada file .faiss di: {model_dir}")
    faiss_filename = faiss_files[0]
    faiss_path     = os.path.join(model_dir, faiss_filename)

    prefix = faiss_filename.replace("_search_index.faiss", "")
    print(f"[FAISS] File ditemukan : {faiss_filename}")
    print(f"[FAISS] Prefix         : '{prefix}'")

    loaded_patchcore.anomaly_scorer.load(model_dir, "")

    print(f"[FAISS] Index berhasil dimuat: {faiss_path}")
    return loaded_patchcore, params


# PREDICTOR
class PatchCorePredictor:
    def __init__(self, model_dir: str, threshold: float = None, device_id: int = 0):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            print(f"[DEVICE] GPU: cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
            print("[DEVICE] CPU")

        self.patchcore, self.params = load_patchcore(model_dir, self.device)

        imagesize = self.params.get("input_shape", (3, 224, 224))[-1]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.threshold = threshold
        if threshold is not None:
            print(f"[THRESHOLD] Manual: {threshold}")
        else:
            print("[THRESHOLD] Belum diset — jalankan --calibrate atau tambah --threshold")

        print("\n[READY] Model siap.\n")

    # ── helper: preprocess gambar ──────────────────────────────────────────────
    def _preprocess(self, image_path: str):
        img_pil    = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        return img_tensor, img_pil

    # ── inferensi satu gambar ──────────────────────────────────────────────────
    def predict_single(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        img_tensor, img_pil = self._preprocess(image_path)

        # Gunakan method predict bawaan PatchCore
        # Return: (scores, masks)
        #   scores : list of image-level anomaly scores
        #   masks  : list of pixel-level anomaly maps
        with torch.no_grad():
            scores, masks = self.patchcore.predict(img_tensor)

        image_score = float(scores[0])
        anomaly_map = masks[0]          # shape: (H, W) numpy array

        threshold   = self.threshold if self.threshold is not None else float("inf")
        verdict     = "SHADOW DEFECT" if image_score > threshold else "OK"

        # Visualisasi
        img_bgr     = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        heatmap     = self._make_heatmap(anomaly_map, img_bgr.shape[:2])
        overlay     = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        overlay     = self._add_text(overlay, verdict, image_score, threshold)

        return {
            "image_path":  image_path,
            "verdict":     verdict,
            "score":       round(image_score, 6),
            "threshold":   threshold,
            "anomaly_map": anomaly_map,
            "overlay":     overlay,
            "heatmap":     heatmap,
        }

    def _make_heatmap(self, anomaly_map: np.ndarray, target_hw: tuple) -> np.ndarray:
        H, W = target_hw
        resized = cv2.resize(anomaly_map.astype(np.float32), (W, H))
        normed  = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normed, cv2.COLORMAP_JET)

    def _add_text(self, img, verdict, score, threshold):
        color  = (0, 0, 255) if verdict == "SHADOW DEFECT" else (0, 200, 0)
        result = img.copy()
        cv2.rectangle(result, (0, 0), (500, 90), (0, 0, 0), -1)
        cv2.putText(result, verdict,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(result,
                    f"Score: {score:.4f}  |  Threshold: {threshold:.4f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1)
        return result

    # ── prediksi batch ─────────────────────────────────────────────────────────
    def predict_batch(self, image_dir: str, save_dir: str = None) -> list:
        files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(SUPPORTED_EXT)
        ])
        if not files:
            print(f"[WARNING] Tidak ada gambar di: {image_dir}")
            return []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        results = []
        for i, path in enumerate(files):
            print(f"[{i+1}/{len(files)}] {os.path.basename(path)}", end=" → ")
            try:
                r = self.predict_single(path)
                print(f"{r['verdict']}  (score: {r['score']:.4f})")
                if save_dir:
                    base = os.path.splitext(os.path.basename(path))[0]
                    cv2.imwrite(os.path.join(save_dir, f"{base}_overlay.jpg"), r["overlay"])
                    cv2.imwrite(os.path.join(save_dir, f"{base}_heatmap.jpg"), r["heatmap"])
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}")

        return results

    # ── kalibrasi threshold ────────────────────────────────────────────────────
    def calibrate(self, normal_dir: str, defect_dir: str, save: bool = True) -> float:
        print("\n[CALIBRATE] Menghitung threshold optimal...")

        def score_dir(folder, label_name):
            files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(SUPPORTED_EXT)
            ])
            print(f"  {label_name}: {len(files)} gambar")
            scores = []
            for path in files:
                r = self.predict_single(path)
                scores.append(r["score"])
                print(f"    {os.path.basename(path):40s} score: {r['score']:.4f}")
            return scores

        normal_scores = score_dir(normal_dir, "Normal")
        defect_scores = score_dir(defect_dir, "Defect")

        labels = [0] * len(normal_scores) + [1] * len(defect_scores)
        scores = normal_scores + defect_scores

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        idx = int(np.argmax(tpr - fpr))
        T   = float(thresholds[idx])

        self.threshold = T

        print(f"\n[CALIBRATE] Hasil:")
        print(f"  AUROC              : {auc:.4f}")
        print(f"  Threshold optimal  : {T:.6f}")
        print(f"  TPR (sensitivity)  : {tpr[idx]:.4f}")
        print(f"  FPR (1-specificity): {fpr[idx]:.4f}")
        print(f"  Max normal score   : {max(normal_scores):.6f}")
        print(f"  Min defect score   : {min(defect_scores):.6f}")

        if save:
            data = {
                "threshold":        T,
                "auroc":            auc,
                "tpr":              float(tpr[idx]),
                "fpr":              float(fpr[idx]),
                "normal_score_max": max(normal_scores),
                "defect_score_min": min(defect_scores),
            }
            with open("threshold.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n[SAVED] threshold.json")

        return T


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="PatchCore Inference — Shadow Defect")
    p.add_argument("--model_dir",   required=True,
                   help="Folder model (berisi .faiss dan .pkl)")
    p.add_argument("--image",       default=None,
                   help="Path satu gambar")
    p.add_argument("--image_dir",   default=None,
                   help="Folder berisi banyak gambar")
    p.add_argument("--save_dir",    default="inference_results",
                   help="Folder output hasil visualisasi")
    p.add_argument("--threshold",   type=float, default=None,
                   help="Threshold manual (jika tidak diisi, cari threshold.json)")
    p.add_argument("--calibrate",   action="store_true",
                   help="Mode kalibrasi threshold")
    p.add_argument("--normal_dir",  default=None,
                   help="(--calibrate) Folder gambar normal")
    p.add_argument("--defect_dir",  default=None,
                   help="(--calibrate) Folder gambar defect")
    p.add_argument("--gpu",         type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # Load threshold dari file jika tidak diisi manual
    threshold = args.threshold
    if threshold is None and os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            threshold = json.load(f).get("threshold")
        print(f"[THRESHOLD] Dimuat dari threshold.json: {threshold:.6f}")

    predictor = PatchCorePredictor(
        model_dir  = args.model_dir,
        threshold  = threshold,
        device_id  = args.gpu,
    )

    # ── MODE: kalibrasi ────────────────────────────────────────────────────────
    if args.calibrate:
        if not args.normal_dir or not args.defect_dir:
            print("[ERROR] --calibrate butuh --normal_dir dan --defect_dir")
            sys.exit(1)
        predictor.calibrate(args.normal_dir, args.defect_dir)
        return

    # ── MODE: satu gambar ──────────────────────────────────────────────────────
    if args.image:
        r = predictor.predict_single(args.image)
        print(f"\n{'='*50}")
        print(f"  File     : {os.path.basename(r['image_path'])}")
        print(f"  Verdict  : {r['verdict']}")
        print(f"  Score    : {r['score']}")
        print(f"  Threshold: {r['threshold']}")
        print(f"{'='*50}\n")
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(os.path.join(args.save_dir, f"{base}_overlay.jpg"), r["overlay"])
        cv2.imwrite(os.path.join(args.save_dir, f"{base}_heatmap.jpg"), r["heatmap"])
        print(f"[SAVED] {args.save_dir}/{base}_overlay.jpg")
        print(f"[SAVED] {args.save_dir}/{base}_heatmap.jpg")
        return

    # ── MODE: batch ────────────────────────────────────────────────────────────
    if args.image_dir:
        results = predictor.predict_batch(args.image_dir, args.save_dir)
        if results:
            n_defect = sum(1 for r in results if r["verdict"] == "SHADOW DEFECT")
            print(f"\n{'='*50}")
            print(f"  Total    : {len(results)}")
            print(f"  OK       : {len(results) - n_defect}")
            print(f"  Defect   : {n_defect}")
            print(f"  Output   : {args.save_dir}/")
            print(f"{'='*50}")
            # Simpan CSV ringkasan
            csv_path = os.path.join(args.save_dir, "summary.csv")
            with open(csv_path, "w") as f:
                f.write("image,verdict,score,threshold\n")
                for r in results:
                    f.write(f"{os.path.basename(r['image_path'])},"
                            f"{r['verdict']},{r['score']},{r['threshold']}\n")
            print(f"[SAVED] {csv_path}")
        return

    print("[ERROR] Tentukan --image, --image_dir, atau --calibrate")
    sys.exit(1)


if __name__ == "__main__":
    main()
    # Untuk debugging cepat tanpa CLI
    # predictor = PatchCorePredictor(
    #     model_dir  = "results/project/models/mvtec_screen",
    #     threshold  = 2.5,
    #     device_id  = 0,
    # )
    # pred = predictor.predict_single("debugging/screen/test/good/000.png")
    # print(pred)