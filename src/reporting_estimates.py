# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
import torch
import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import List, Dict, Optional

def extract_betas(model, config, refmode=0, alt_names=None):
    """
    Extracts the betas (parameters) from the trained model for reporting purposes.
    Returns the **constrained** betas used in the forward pass, not the raw, unconstrained betas.
    Assumes the reference mode is the first alternative; see dcm_seal_model.py full_asc definitions
    """
    J=config['n_alternatives']
    C = len(config["core_vars"])
    E = len(config["embedding_vars"])
    K = config['n_latent_classes']
    if not alt_names:
        alt_names=list(range(1,J+1))

    if not (0<=refmode<J):
        raise ValueError(f"reference_alt index {refmode} out of range for [0,J-1]; J: {J}.")

    betas = model.beta  # Get constrained betas (model.beta already applies softplus and non-positivity constraints)
    beta_list = []

    # ---- Prepend ASC columns from model.asc (shape: (K, J-1) or (J-1,)) ----
    asc_cols = []
    asc_col_names = []
    
    if hasattr(model, "asc"):
        asc = model.asc
        if isinstance(asc, torch.nn.Parameter):
            asc = asc.data
        if asc.ndim == 1:
            asc = asc.unsqueeze(0)  # (J-1,) -> (1, J-1) for K=1
        asc_np = asc.detach().cpu().numpy()  # (K, J-1)
        
        if refmode == J - 1:
            nonref_indices = list(range(0, J - 1))             # last is ref → keep alts 0..J-2
        elif refmode == 0:
            nonref_indices = list(range(1, J))                 # first is ref → keep alts 1..J-1
        else:
            # (General case) If you ever use a middle reference, align to your forward-pass ordering here.
            # By convention, put all alts left of ref, then all right of ref (matching how you concatenate).
            nonref_indices = list(range(0, refmode)) + list(range(refmode + 1, J))
        asc_col_names = [f"ASC_{alt_names[j]}" for j in nonref_indices]
        asc_cols = [asc_np[:, c] for c in range(asc_np.shape[1])]
    else:
        asc_np = None  # no ASCs

    beta_list = []
    col_names = []

    if asc_cols:
        beta_list.extend(asc_cols)
        col_names.extend(asc_col_names)
    
    if C > 0:
        core_betas = betas[:, :C] if betas.ndim == 2 else betas[:C].unsqueeze(0)
        for i, name in enumerate(config["core_vars"]):
            beta_list.append(core_betas[:, i].detach().cpu().numpy())
            col_names.append(name)

    if E > 0:
        emb_betas = betas[:, C:C+E] if betas.ndim == 2 else betas[C:C+E].unsqueeze(0)
        for i, name in enumerate(config["embedding_vars"]):
            beta_list.append(emb_betas[:, i].detach().cpu().numpy())
            col_names.append(name)

    # ---- Assemble DataFrame ----
    if len(beta_list) == 0:
        beta_df = pd.DataFrame(index=np.arange(K))
    else:
        beta_df = pd.DataFrame(np.column_stack(beta_list), columns=col_names)

    # Optionally include the raw betas as well (for debugging purposes)
    beta_raw = model.beta_raw  # This gives unconstrained betas if needed

    # Return the dictionary containing the betas as numpy arrays
    return beta_df, beta_raw

def _to_tensor(x) -> torch.Tensor:
    w = x.weight if hasattr(x, "weight") else x
    if isinstance(w, torch.nn.Parameter):
        w = w.data
    return w.detach().cpu()


def _row_labels_from_hparams(hparams: dict, key: str = "embedding_labels") -> List[str]:
    """
    Flatten the (count, labels) blocks from hparams[key] (an OrderedDict):
      { var: (count, [labels...]), ... }  -->  [labels...] in block order
    Assumes labels are already human-readable and in training order.
    """
    emb = hparams.get(key, None)
    if emb is None:
        raise KeyError(f"hparams['{key}'] not found.")
    rows: List[str] = []
    for var, (cnt, labels) in emb.items():
        if not isinstance(labels, (list, tuple)) or len(labels) != cnt:
            # fallback if something is off in the entry
            labels = [f"{var}[{i}]" for i in range(cnt)]
        rows.extend(map(str, labels))
    return rows


def extract_embedding(model,alt_names: Optional[List[str]] = None,hparams_key: str = "embedding_labels") -> Dict[str, pd.DataFrame]:
    """
    Return {'global': df} for shared embeddings, or {'class_1': df, ..., 'class_K': df}
    for class-specific embeddings. Each df mirrors the embedding matrix shape (n_levels, n_alts).
    Row labels come directly from model.hparams[hparams_key] (always available per your setup).
    """
    # 1) locate embedding container
    emb = getattr(model, "embedding_layers", None)
    if emb is None and isinstance(model, Mapping):
        emb = model.get("embedding_layers", None)
    if emb is None:
        raise AttributeError("embedding_layers not found on model (attr or mapping key).")

    # 2) build row labels from hparams (single source of truth)
    row_labels = _row_labels_from_hparams(model.hparams, key=hparams_key)

    def make_df(W: torch.Tensor, tag: str) -> pd.DataFrame:
        if W.ndim != 2:
            raise ValueError(f"{tag}: expected 2D (n_levels, n_alts), got {tuple(W.shape)}")
        n_levels, n_alts = W.shape

        rows = row_labels if len(row_labels) == n_levels else [f"level_{i}" for i in range(n_levels)]
        cols = alt_names if (alt_names and len(alt_names) == n_alts) else [f"alt_{j+1}" for j in range(n_alts)]

        return pd.DataFrame(W.numpy(), index=rows, columns=cols)

    # 3) emit one DataFrame per class, or one 'global'
    tables: Dict[str, pd.DataFrame] = {}
    if isinstance(emb, torch.nn.ModuleList):
        for k, item in enumerate(emb):
            W = _to_tensor(item)
            tables[f"class_{k+1}"] = make_df(W, f"class_{k+1}")
    else:
        W = _to_tensor(emb)
        tables["global"] = make_df(W, "global")
    return tables

@torch.no_grad()
def predict_membership(model, df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    """
    Use a trained model's segmentation network to get class-membership probabilities
    from a pandas DataFrame that already contains the converted columns listed in
    model.hparams['segmentation_vars'].

    Returns a DataFrame with rows = input ids (or 0..N-1) and columns = class_1..class_K.
    """
    model.eval()
    device = next(model.parameters()).device

    K = int(getattr(model.hparams, "n_latent_classes", 1))
    seg_vars = list(getattr(model.hparams, "segmentation_vars", []) or [])

    # If K==1, no segmentation net is used; return a single column of ones.
    if K == 1 or not hasattr(model, "segmentation_net") or model.segmentation_net is None:
        ids = df[id_col].tolist() if id_col and id_col in df.columns else list(range(len(df)))
        out = pd.DataFrame({"class_1": [1.0] * len(ids)}, index=ids)
        out.index.name = id_col if id_col else "index"
        return out

    # Ensure all required columns exist; add missing as zeros, keep exact order.
    X = pd.DataFrame(index=df.index)
    for c in seg_vars:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = 0.0  # unseen indicator -> 0

    # Convert to tensor
    x_seg = torch.as_tensor(X.values, dtype=torch.float32, device=device)

    # Forward through segmentation net -> logits (N, K)
    logits = model.segmentation_net(x_seg)
    if logits.ndim == 1:  # safety
        logits = logits.unsqueeze(0)

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    cols = [f"class_{i+1}" for i in range(K)]

    # Row index: use id_col if provided, else default index
    if id_col and id_col in df.columns:
        idx = df[id_col].tolist()
        idx_name = id_col
    else:
        idx = df.index.tolist()
        idx_name = df.index.name or "index"

    out = pd.DataFrame(probs, index=idx, columns=cols)
    out.index.name = idx_name
    return out

if __name__ == '__main__':
    pass
