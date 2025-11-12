# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PaddedChoiceDataset(Dataset):
    """
    Precomputes per-chid tensors once so __getitem__ is O(1) and the default collate can just stack. Shapes:
      - core_bank:  (N, J_max, C)
      - mask_bank:  (N, J_max)              bool
      - choice_bank:(N,)                    long
      - seg_bank:   (N, S)                  float32
      - x_emb_bank: (N, E)                  long
    Legend: N=chids, J_max=max #alts per chid, C=#core vars, S=#seg vars, E=#embedding vars
    Args:
        dataframe: Long-format choice data sorted by chid.
        x_emb_tensor: Embedding indices aligned with ``dataframe`` rows.
        config: Experiment configuration dictionary.
        fast_mode: If True, ``__getitem__`` only returns indices and ``collate_indices``
            can be used to gather whole batches in a single vectorized call, greatly
            reducing Python overhead during data loading.
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict, *, fast_mode: bool = False):
        super().__init__()

        # Define core, segmentation, and embedding variables
        self.core_vars = config.get("core_vars", [])
        self.seg_vars = config.get("segmentation_vars", [])
        self.emb_vars = config.get("embedding_vars", [])
        self.J_max = config["n_alternatives"]  # (max J)
        self.fast_mode = fast_mode

        if len(self.core_vars) == 0:
            raise ValueError("config['core_vars'] is empty; the model expects at least one core variable.")
        if 'chid' not in dataframe.columns or 'alt' not in dataframe.columns or 'match' not in dataframe.columns:
            raise ValueError("DataFrame must contain columns: 'chid', 'alt', 'match'.")

        if x_emb_tensor.size(0) != len(dataframe):
            raise ValueError("x_emb_tensor must have the same number of rows as the dataframe.")

        # --- factorize once to obtain contiguous chid codes preserving encounter order ---
        codes, uniques = pd.factorize(dataframe['chid'], sort=False)
        if (codes < 0).any():
            raise ValueError("All rows must have a valid 'chid' value.")
        self.chids = uniques.to_numpy()
        N = len(self.chids)

        # --- locate the first row of each chid (for seg/x_emb banks) ---
        if len(codes) == 0:
            first_idx = np.empty((0,), dtype=np.int64)
        else:
            first_marker = np.empty_like(codes, dtype=bool)
            first_marker[0] = True
            np.not_equal(codes[1:], codes[:-1], out=first_marker[1:])
            first_idx = np.nonzero(first_marker)[0]

        if len(first_idx) != N:
            raise RuntimeError("Failed to identify one representative row per 'chid'.")

        # --- banks with one row per chid ---
        if self.seg_vars:
            seg_np = dataframe.iloc[first_idx][self.seg_vars].to_numpy(dtype=np.float32, copy=False)
            self.seg_bank = torch.from_numpy(seg_np)
        else:
            self.seg_bank = torch.empty((N, 0), dtype=torch.float32)
        E = x_emb_tensor.shape[1] if x_emb_tensor.ndim == 2 else 0
        if E == 0:
            self.x_emb_bank = x_emb_tensor.new_zeros((N, 0))
        else:
            self.x_emb_bank = x_emb_tensor[first_idx] if len(first_idx) else x_emb_tensor.new_zeros((0, E))

        # --- compute per-alternative placement indices ---
        alt_position = dataframe.groupby('chid', sort=False).cumcount().to_numpy(dtype=np.int64)

        counts = np.bincount(codes, minlength=N)
        max_count = counts.max() if counts.size else 0
        if max_count > self.J_max:
            offender = self.chids[counts.argmax()]
            raise ValueError(f"Found chid={offender} with {max_count} alternatives > J_max={self.J_max}.")

        C = len(self.core_vars)
        core_bank_np = np.zeros((N, self.J_max, C), dtype=np.float32)
        core_values = dataframe[self.core_vars].to_numpy(dtype=np.float32, copy=False)
        core_bank_np[codes, alt_position, :] = core_values

        mask_bank_np = np.zeros((N, self.J_max), dtype=bool)
        mask_bank_np[codes, alt_position] = True

        choice_bank_np = np.zeros(N, dtype=np.int64)
        match_mask = dataframe['match'].to_numpy(dtype=bool, copy=False)
        choice_bank_np[codes[match_mask]] = alt_position[match_mask]

        self.core_bank = torch.from_numpy(core_bank_np)
        self.mask_bank = torch.from_numpy(mask_bank_np)
        self.choice_bank = torch.from_numpy(choice_bank_np)

    def __len__(self):
        return len(self.chids)

    def __getitem__(self, idx: int):
        if self.fast_mode:
            return int(idx)

        # Tensors are precomputed during init. this method only slices
        return {
            'core_features': self.core_bank[idx],  # (J, C)
            'seg_features': self.seg_bank[idx],    # (S,)
            'x_emb': self.x_emb_bank[idx],         # (E,)
            'mask': self.mask_bank[idx],           # (J,)
            'choice': self.choice_bank[idx],       # ()
        }

    def _batch_from_indices(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Vectorized gather of multiple chids at once."""
        if idx.numel() == 0:
            raise ValueError("Received an empty batch of indices.")

        if idx.dtype != torch.long:
            idx = idx.to(torch.long)

        return {
            'core_features': self.core_bank.index_select(0, idx),
            'seg_features': self.seg_bank.index_select(0, idx),
            'x_emb': self.x_emb_bank.index_select(0, idx),
            'mask': self.mask_bank.index_select(0, idx),
            'choice': self.choice_bank.index_select(0, idx),
        }

    def collate_indices(self, batch: list[int]) -> dict[str, torch.Tensor]:
        """Efficient collate_fn that batches using vectorized indexing."""
        if not self.fast_mode:
            raise RuntimeError("collate_indices is only valid when fast_mode=True")

        idx = torch.as_tensor(batch, dtype=torch.long)
        return self._batch_from_indices(idx)


if __name__ == '__main__':
    pass