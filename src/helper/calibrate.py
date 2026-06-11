from typing import Literal
import os
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
from loguru import logger
from tqdm import tqdm

EXTS = [".png", ".jpg", ".pneg"]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def score_dir(model, folder):
    fname = Path(folder).name
    paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in EXTS
    ])
    if not paths:
        raise ValueError(f"No found image in {folder}")
    
    scores = []
    for p in tqdm(paths, f'Scoring {fname}'):
        img = Image.open(p)
        img.load()
        tensor = DEFAULT_TRANSFORM(img).unsqueeze(0)

        r = model.predict_tensor(tensor)
        scores.append(r['score'])
    return scores

def find(model, good_dir: str, defect_dir: str = None, method: Literal['percentile', 'youden'] = 'youden', percentile: float = 85.0) -> dict:
    good_scores = score_dir(model, good_dir)

    if method == "percentile":
        threshold = float(np.percentile(good_scores, percentile))
        logger.info(f"Thereshold (percentile={percentile:.1f}): {threshold:.4f}")
        return best_thresh
            
    defect_scores = score_dir(model, defect_dir)
    scores = np.array(good_scores + defect_scores, np.float32)
    labels = np.array([0] * len(good_scores) + [1] * len(defect_scores), dtype=np.int32)

    # AUROC
    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    auroc = float(np.trapz(tpr, fpr))

    candidates = np.unique(scores)
    best_thresh = candidates[0]
    best_metric = -1
    best_f1 = 0
    best_sens = 0
    best_spec = 0

    for thresh in candidates:
        pred = (scores >= thresh).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
 
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
 
        metric = (sens + spec - 1) if method == "youden" else f1
 
        if metric > best_metric:
            best_metric = metric
            best_thresh = float(thresh)
            best_f1 = f1
            best_sens = sens
            best_spec = spec
 
    logger.debug(
        f"Threshold optimal {best_thresh:.4f}  AUROC={auroc:.3f}  F1={best_f1:.3f}  sens={best_sens:.3f}  spec={best_spec:.3f}"
    )
 
    return best_thresh