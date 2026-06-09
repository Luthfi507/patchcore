import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import PIL.Image
import PIL.ImageFile

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    project_folder, group_folder
):
    os.makedirs("result", exist_ok=True)
    project_path = os.path.join("result", project_folder)
    
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)

    counter = 0
    while os.path.exists(save_path):
        save_path = os.path.join(project_path, group_folder + "_" + str(counter))
        counter += 1
    os.makedirs(save_path)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

def check_one(path: str) -> dict:
    result = {
        "path": path,
        "ok": True,
        "error": None,
        "width": None,
        "height": None,
        "mode": None,
        "filesize_mb": None,
    }
 
    try:
        result["filesize_mb"] = os.path.getsize(path) / 1024 / 1024
 
        if result["filesize_mb"] == 0:
            result["ok"] = False
            result["error"] = "EMPTY_FILE"
            return result
 
        # Buka header saja (cepat, tidak decode pixel)
        with PIL.Image.open(path) as img:
            result["width"]  = img.width
            result["height"] = img.height
            result["mode"]   = img.mode
            # Decode pixel penuh — ini yang bisa hang kalau ada masalah
            img.convert("RGB")
            img.load()
 
    except Exception as e:
        result["ok"]    = False
        result["error"] = f"{type(e).__name__}: {e}"
 
    return result
 
 
def collect_all_images(data_path: str) -> list:
    paths = []
    for root, dirs, files in os.walk(data_path):
        dirs.sort()
        for fname in sorted(files):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(root, fname))
    return paths
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--workers",   type=int, default=8)
    parser.add_argument("--output",    type=str, default="corrupt_images.txt")
    args = parser.parse_args()
 
    print(f"Scanning: {args.data_path}")
    all_paths = collect_all_images(args.data_path)
    total = len(all_paths)
    print(f"Total gambar: {total}\n")
 
    if total == 0:
        print("Tidak ada gambar ditemukan.")
        sys.exit(1)
 
    results  = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, p): p for p in all_paths}
        done = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            results.append(r)
            if not r["ok"]:
                print(f"  [ERROR] {r['path']}\n         → {r['error']}")
            if done % 200 == 0 or done == total:
                print(f"  Progress: {done}/{total}", end="\r")
 
    print(f"\n\n{'='*60}")
 
    # ── Statistik resolusi ────────────────────────────────────────────────
    ok_results = [r for r in results if r["ok"]]
    bad_results = [r for r in results if not r["ok"]]
 
    if ok_results:
        widths  = [r["width"]  for r in ok_results]
        heights = [r["height"] for r in ok_results]
        fsizes  = [r["filesize_mb"] for r in ok_results]
        modes   = {}
        for r in ok_results:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
 
        print(f"STATISTIK DATASET:")
        print(f"  Total gambar   : {total}")
        print(f"  OK             : {len(ok_results)}")
        print(f"  Error/Corrupt  : {len(bad_results)}")
        print(f"")
        print(f"  Resolusi:")
        print(f"    Min  : {min(widths)} x {min(heights)}")
        print(f"    Max  : {max(widths)} x {max(heights)}")
        print(f"    Mean : {sum(widths)//len(widths)} x {sum(heights)//len(heights)}")
        print(f"")
        print(f"  Ukuran file:")
        print(f"    Min  : {min(fsizes):.2f} MB")
        print(f"    Max  : {max(fsizes):.2f} MB")
        print(f"    Mean : {sum(fsizes)/len(fsizes):.2f} MB")
        print(f"    Total: {sum(fsizes):.1f} MB")
        print(f"")
        print(f"  Mode warna: {modes}")
 
        # Flagging gambar yang sangat besar (> 10 MB atau > 4000px)
        large = [r for r in ok_results
                 if r["filesize_mb"] > 10 or r["width"] > 4000 or r["height"] > 4000]
        if large:
            print(f"\n  [!] Gambar berukuran sangat besar ({len(large)} file):")
            for r in sorted(large, key=lambda x: -x["filesize_mb"])[:20]:
                print(f"      {r['path']}")
                print(f"        {r['width']}x{r['height']}, {r['filesize_mb']:.1f} MB")
 
    print(f"{'='*60}")
 
    # ── Simpan daftar error ───────────────────────────────────────────────
    if bad_results:
        with open(args.output, "w") as f:
            for r in sorted(bad_results, key=lambda x: x["path"]):
                f.write(f"{r['path']}\t{r['error']}\n")
        print(f"\nFile bermasalah disimpan ke: {args.output}")
    else:
        print(f"\nSemua gambar dapat dibaca.")
 
        # Kalau tidak ada yang corrupt tapi training tetap hang,
        # kemungkinan masalahnya bukan di file
        print(f"\nKarena tidak ada gambar corrupt, hang kemungkinan disebabkan:")
        print(f"  1. to_tensor() lambat pada gambar resolusi sangat tinggi")
        print(f"     → Pastikan resolusi Max di atas tidak terlalu besar")
        print(f"  2. FAISS memory issue saat coreset sampling")
        print(f"     → Coba tambah --coreset_percentage 0.01")
        print(f"  3. Memory leak di feature aggregation loop")
        print(f"     → Coba kurangi --batch_size ke 2")
 
 
if __name__ == "__main__":
    main()