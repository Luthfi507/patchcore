import os

import PIL.Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split="train",
        train_val_split=1.0
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname]
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)
        # Simpan mean/std agar run_patchcore bisa denormalize gambar saat visualisasi
        self.transform_mean = IMAGENET_MEAN
        self.transform_std  = IMAGENET_STD

    @staticmethod
    def _safe_open(path, mode="RGB"):
        """
        Buka gambar dan paksa decode pixel sekarang (bukan lazy).

        PIL.Image.open() secara default adalah lazy — pixel baru di-decode
        saat pertama kali diakses (misalnya di to_tensor()). Kalau ada
        masalah di file (truncated, encoding aneh, EXIF besar), decode
        bisa hang tanpa error di to_tensor().

        Solusi: panggil .load() segera setelah open() agar decode terjadi
        di sini, di dalam try/except, bukan di dalam transform pipeline.
        """
        img = PIL.Image.open(path)
        img.load()          # paksa decode pixel sekarang — tidak lazy lagi
        return img.convert(mode)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        # ── Load & decode image sekarang (bukan lazy) ────────────────────
        try:
            image = self._safe_open(image_path, "RGB")
            image = self.transform_img(image)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Gambar tidak bisa dibaca (idx=%d): %s — %s — diganti blank.",
                idx, image_path, e,
            )
            blank = PIL.Image.new("RGB", (self.imagesize[1], self.imagesize[2]), color=0)
            image = self.transform_img(blank)

        if self.split == "test" and mask_path is not None:
            try:
                mask = self._safe_open(mask_path, "L")
                mask = self.transform_mask(mask)
            except Exception:
                mask = torch.zeros([1, *image.size()[1:]])
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _split_image_paths(self, image_paths):
        if self.train_val_split >= 1.0:
            return image_paths

        split_idx = int(len(image_paths) * self.train_val_split)
        if self.split == "train":
            return image_paths[:split_idx]
        if self.split == "val":
            return image_paths[split_idx:]
        return image_paths

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = sorted(os.listdir(classpath))

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                image_paths = [os.path.join(anomaly_path, x) for x in anomaly_files]
                image_paths = self._split_image_paths(image_paths)
                imgpaths_per_class[classname][anomaly] = image_paths

                if self.split == "test" and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname][anomaly] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                image_paths = imgpaths_per_class[classname][anomaly]
                mask_paths = maskpaths_per_class[classname][anomaly]
                for i, image_path in enumerate(image_paths):
                    data_to_iterate.append(
                        [
                            classname,
                            anomaly,
                            image_path,
                            mask_paths[i] if mask_paths is not None else None,
                        ]
                    )

        return imgpaths_per_class, data_to_iterate