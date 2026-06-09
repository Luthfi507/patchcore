import abc
import os
from typing import Union

import numpy as np
import torch
from tqdm import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage <= 1:
            raise ValueError("Percentage value must be in (0, 1].")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)

    def _to_tensor(self, features: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        self._store_type(features)
        if isinstance(features, np.ndarray):
            return torch.from_numpy(features)
        return features


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features.to(self.device)
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        ).to(self.device)
        return mapper(features.to(self.device))

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.percentage == 1:
            return features
        features = self._to_tensor(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = distance_matrix.norm(dim=1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)
        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)
            coreset_select_distance = distance_matrix[:, select_idx : select_idx + 1]
            coreset_anchor_distances = torch.min(
                torch.cat(
                    [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance],
                    dim=1,
                ),
                dim=1,
            ).values
        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        num_workers: int = 0,
        vram_budget_gb: float = 2.0,
        use_gpu_if_available: bool = True,
    ):
        """Approximate Greedy Coreset sampling — GPU-aware, anti-OOM.

        Strategi memori:
          Cek VRAM bebas saat run() dipanggil, lalu tentukan berapa banyak
          features yang bisa di-load ke GPU sekaligus.

          • Jika semua features muat → semuanya di GPU, loop tanpa transfer.
          • Jika tidak muat → bagi jadi shard sebesar mungkin. Tiap shard
            di-load ke GPU SEKALI per-iterasi (bukan per-chunk), lalu di-free.
            Ini jauh lebih cepat dari chunking kecil-kecil karena jumlah
            transfer = num_shards × num_iterations (bukan N/chunk × iterations).

        Args:
            percentage: Fraksi fitur yang diambil.
            device: Device untuk backbone (tempat features masuk).
            number_of_starting_points: Jumlah titik awal aproksimasi.
            dimension_to_project_features_to: Dimensi proyeksi sebelum sampling.
            num_workers: CPU threads untuk matmul fallback. 0 = semua core.
            vram_budget_gb: VRAM yang disisakan untuk backbone/inference (GB).
                            Features akan dibagi shard agar tidak melebihi
                            sisa VRAM - vram_budget_gb.
            use_gpu_if_available: Gunakan GPU jika tersedia.
        """
        self.number_of_starting_points = number_of_starting_points
        self.use_gpu_if_available = use_gpu_if_available
        self.vram_budget_gb = vram_budget_gb
        self._n_threads = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _get_compute_device(self) -> torch.device:
        if self.use_gpu_if_available and torch.cuda.is_available():
            if hasattr(self, 'device') and self.device.type == 'cuda':
                return self.device
            return torch.device('cuda:0')
        return torch.device('cpu')

    def _max_rows_on_gpu(self, features_cpu: torch.Tensor, compute_device: torch.device) -> int:
        """Hitung berapa baris features yang bisa di-load ke GPU sekaligus.

        Sisakan vram_budget_gb untuk backbone dan operasi lain.
        """
        if compute_device.type != 'cuda':
            return len(features_cpu)  # CPU: tidak ada batasan VRAM

        bytes_per_row = features_cpu.shape[1] * features_cpu.element_size()

        # VRAM bebas saat ini
        free_vram, total_vram = torch.cuda.mem_get_info(compute_device)
        budget_bytes = free_vram - int(self.vram_budget_gb * 1024 ** 3)
        budget_bytes = max(budget_bytes, 0)

        if budget_bytes <= 0:
            return 0  # tidak ada ruang sama sekali → fallback CPU

        max_rows = budget_bytes // bytes_per_row
        return int(max_rows)

    @staticmethod
    def _dist_sq_norm(
        feat: torch.Tensor,       # [N x D]
        feat_norms_sq: torch.Tensor,  # [N x 1]
        query: torch.Tensor,      # [M x D]
    ) -> torch.Tensor:
        """||a - b||² = ||a||² - 2<a,b> + ||b||²  →  distance [N x M]"""
        q_norms_sq = (query * query).sum(dim=1, keepdim=True).T  # [1 x M]
        return (-2.0 * feat.mm(query.T) + feat_norms_sq + q_norms_sq).clamp(0).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self._n_threads)
        try:
            # Selalu mulai dari CPU — _reduce_features bisa meninggalkan di GPU
            return self._run_greedy_loop(features.cpu())
        finally:
            torch.set_num_threads(prev_threads)

    def _run_greedy_loop(self, features: torch.Tensor) -> np.ndarray:
        """
        Jalankan greedy coreset di GPU sebanyak yang muat, sisanya di CPU.

        Jika semua features muat di VRAM → zero-transfer loop (paling cepat).
        Jika tidak muat → bagi jadi shard besar; tiap shard di-upload ke GPU
        SEKALI per iterasi, bukan per-chunk kecil.
        """
        compute_device = self._get_compute_device()
        N = len(features)

        max_rows = self._max_rows_on_gpu(features, compute_device)

        if compute_device.type == 'cpu' or max_rows <= 0:
            # Fallback: semua di CPU dengan multi-thread matmul
            return self._greedy_loop_sharded(features, features, compute_device)

        if max_rows >= N:
            # Happy path: semua muat di GPU → upload sekali, loop tanpa transfer
            feat_gpu = features.to(compute_device)
            return self._greedy_loop_sharded(features, feat_gpu, compute_device)
        else:
            # Partial fit: bagi jadi shard sebesar max_rows
            # Upload tiap shard ke GPU SEKALI per-iterasi
            return self._greedy_loop_sharded_partial(features, max_rows, compute_device)

    def _greedy_loop_sharded(
        self,
        features_cpu: torch.Tensor,   # [N x D] di CPU — untuk indexing
        features_dev: torch.Tensor,   # [N x D] di compute_device — untuk komputasi
        compute_device: torch.device,
    ) -> np.ndarray:
        """Loop greedy dengan features sepenuhnya di compute_device (CPU atau GPU)."""
        N = len(features_dev)

        # Precompute norms sekali
        with torch.no_grad():
            norms_sq = (features_dev * features_dev).sum(dim=1, keepdim=True)  # [N x 1]

        # Jarak awal ke starting points
        n_start = min(self.number_of_starting_points, N)
        start_idx = np.random.choice(N, n_start, replace=False)
        start_pts = features_dev[start_idx]  # [n_start x D]

        with torch.no_grad():
            init_dists = self._dist_sq_norm(features_dev, norms_sq, start_pts)  # [N x n_start]
            anchor_dists = init_dists.mean(dim=1, keepdim=True).cpu()  # [N x 1] CPU

        coreset_indices = []
        n_samples = int(N * self.percentage)

        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="Subsampling..."):
                select_idx = torch.argmax(anchor_dists).item()
                coreset_indices.append(select_idx)

                query = features_dev[select_idx : select_idx + 1]  # [1 x D]
                q_norm_sq = norms_sq[select_idx : select_idx + 1]  # [1 x 1]
                new_dists = self._dist_sq_norm(features_dev, norms_sq, query).cpu()  # [N x 1]

                anchor_dists = torch.min(
                    torch.cat([anchor_dists, new_dists], dim=1), dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

    def _greedy_loop_sharded_partial(
        self,
        features_cpu: torch.Tensor,  # [N x D] di CPU
        max_rows: int,               # baris maks yang muat di GPU
        compute_device: torch.device,
    ) -> np.ndarray:
        """Loop greedy ketika features tidak muat semua di GPU.

        Bagi features jadi shard [max_rows x D]. Tiap iterasi:
          1. Untuk tiap shard: upload ke GPU, hitung jarak, turun ke CPU, free.
          2. Gabungkan hasil jarak dari semua shard di CPU.
          3. Update anchor_dists di CPU.

        Transfer per-iterasi = ceil(N/max_rows) upload+download (bukan N/chunk kecil).
        Lebih cepat jika max_rows besar (mendekati N).
        """
        N = len(features_cpu)
        n_shards = max(1, int(np.ceil(N / max_rows)))

        # Precompute norms di CPU
        with torch.no_grad():
            norms_sq_cpu = (features_cpu * features_cpu).sum(dim=1, keepdim=True)

        # Jarak awal ke starting points
        n_start = min(self.number_of_starting_points, N)
        start_idx = np.random.choice(N, n_start, replace=False)

        with torch.no_grad():
            # Hitung jarak ke starting points per-shard
            init_chunks = []
            for s in range(n_shards):
                r0, r1 = s * max_rows, min((s + 1) * max_rows, N)
                f_gpu = features_cpu[r0:r1].to(compute_device)
                n_gpu = norms_sq_cpu[r0:r1].to(compute_device)
                sp_gpu = features_cpu[start_idx].to(compute_device)
                d = self._dist_sq_norm(f_gpu, n_gpu, sp_gpu)
                init_chunks.append(d.cpu())
                del f_gpu, n_gpu, sp_gpu, d
            init_dists = torch.cat(init_chunks, dim=0)  # [N x n_start]
            anchor_dists = init_dists.mean(dim=1, keepdim=True)  # [N x 1] CPU

        coreset_indices = []
        n_samples = int(N * self.percentage)

        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="Subsampling..."):
                select_idx = torch.argmax(anchor_dists).item()
                coreset_indices.append(select_idx)

                # Query vector ke GPU
                query_gpu = features_cpu[select_idx : select_idx + 1].to(compute_device)
                q_norm_gpu = norms_sq_cpu[select_idx : select_idx + 1].to(compute_device)

                # Hitung jarak per-shard
                dist_chunks = []
                for s in range(n_shards):
                    r0, r1 = s * max_rows, min((s + 1) * max_rows, N)
                    f_gpu = features_cpu[r0:r1].to(compute_device)
                    n_gpu = norms_sq_cpu[r0:r1].to(compute_device)
                    d = self._dist_sq_norm(f_gpu, n_gpu, query_gpu)
                    dist_chunks.append(d.cpu())
                    del f_gpu, n_gpu, d
                new_dists = torch.cat(dist_chunks, dim=0)  # [N x 1]

                anchor_dists = torch.min(
                    torch.cat([anchor_dists, new_dists], dim=1), dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        subset_size = int(len(features) * self.percentage)
        if isinstance(features, np.ndarray):
            indices = np.random.choice(len(features), subset_size, replace=False)
            return features[indices]
        indices = torch.randperm(len(features), device=features.device)[:subset_size]
        return features[indices]