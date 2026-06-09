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

def check_one(path: str) -> tuple[str, str | None]:
    try:
        if not os.path.exists(path):
            return path, "FILE_NOT_FOUND"
        
        if os.path.getsize(path) == 0:
            return path, "EMPTY_FILE"
        
        with PIL.Image.open(path) as img:
            img.verify()

        with PIL.Image.open(path) as img:
            img.convert("RGB")
            _ = img.size

        return path, None
    
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"
    
def collect_all_images(data_path: str) -> list[str]:
    paths = []
    for root, dirs, files in os.walk(data_path):
        dirs.sort()
        for fname in sorted(files):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, fname))

    return paths

def main():
    parser = argparse.ArgumentParser(description="Cek gambar corrupt di dataset")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder dataset (misal: dataset/)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Jumlah thread paralel untuk pengecekan")
    parser.add_argument("--output", type=str, default="corrupt_images.txt",
                        help="File output untuk daftar gambar corrupt")
    args = parser.parse_args()
 
    print(f"Scanning: {args.data_path}")
    all_paths = collect_all_images(args.data_path)
    total = len(all_paths)
    print(f"Total gambar ditemukan: {total}")
 
    if total == 0:
        print("Tidak ada gambar ditemukan. Cek --data_path.")
        sys.exit(1)
 
    corrupt = []
    ok_count = 0
 
    print(f"\nMemeriksa {total} gambar dengan {args.workers} thread...\n")
 
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, p): p for p in all_paths}
        done = 0
        for future in as_completed(futures):
            done += 1
            path, error = future.result()
            if error:
                corrupt.append((path, error))
                print(f"  [CORRUPT] {path}\n           → {error}")
            else:
                ok_count += 1
 
            # Progress setiap 100 file
            if done % 100 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({100*done//total}%)", end="\r")
 
    print(f"\n\n{'='*60}")
    print(f"HASIL:")
    print(f"  OK      : {ok_count}")
    print(f"  Corrupt : {len(corrupt)}")
    print(f"{'='*60}")
 
    if corrupt:
        print(f"\nFile corrupt disimpan ke: {args.output}")
        with open(args.output, "w") as f:
            for path, err in sorted(corrupt):
                f.write(f"{path}\t{err}\n")
 
        print("\nDaftar file corrupt:")
        for path, err in sorted(corrupt):
            print(f"  {path}")
            print(f"    {err}")
    else:
        print("\nSemua gambar OK — tidak ada yang corrupt.")
        print("Kemungkinan masalah hang bukan dari file gambar.")
        print("Coba jalankan dengan PYTHONFAULTHANDLER=1 untuk lihat stack trace:")
        print("  PYTHONFAULTHANDLER=1 python run_patchcore.py ... &")
        print("  sleep 200 && kill -SIGSEGV <pid>   # kirim signal saat stuck")
 
if __name__ == "__main__":
    main()