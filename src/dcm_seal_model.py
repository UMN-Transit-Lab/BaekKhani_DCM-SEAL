# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict #needed for config's embedding dims and offset calculation
import pandas as pd
import numpy as np

@torch.no_grad()
def _ll_from_utilities(utilities: torch.Tensor, y_idx: torch.Tensor, row_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum log-likelihood over rows given utilities logits and chosen alt indices."""
    logp = torch.log_softmax(utilities, dim=-1) # (B, J) softmax over J -> still (B, J)
    ll = logp.gather(1, y_idx.view(-1, 1)).squeeze(1) # gather(1:dim,index:(B,1))-> squeeze to (B,)
    if row_mask is not None:
        ll = ll[row_mask]
    return ll.sum()

@torch.no_grad()
def _ll0_uniform(mask: torch.Tensor | None, batch_size: int, n_alts: int, device: torch.device) -> torch.Tensor:
    """Null LL under uniform over AVAILABLE alts per row."""
    if mask is None: 
        k = torch.full((batch_size,), n_alts, device=device, dtype=torch.float32)
    else: # mask is (B,J)
        k = mask.float().sum(dim=1).clamp_min(1) # sum makes (B,), clamp_min sets lower bound
    return -k.log().sum()

def _rho2(ll_hat: torch.Tensor, ll0: torch.Tensor) -> torch.Tensor:
    return 1.0 - (ll_hat / ll0)


class DCM_SEAL(pl.LightningModule):
    def __init__(self, config: dict):
        """
        Initializes the DCM-SEAL model with a global embedding matrix architecture.

        Args:
            config (dict): A dictionary containing model hyperparameters; defined in main.py
                Expected keys:
                - 'n_latent_classes' (int): Number of latent classes.
                - 'n_alternatives' (int): Number of maximum choice alternatives.
                - 'choice_mode' (str): 'heterogeneous' or 'homogeneous'.
                - 'embedding_mode' (str): 'shared' or 'class-specific'.
                - 'embedding_dims' (dict): An *ordered* dict mapping categorical var names
                  to their # of unique categories. e.g., {"purpose": 10, "hr": 24, ...}
                - 'segmentation_vars' (list[str]): Column names for segmentation variables.
                - 'core_vars' (list[str]): Column names for core utility variables.
                - 'non_positive_core_vars (list[str]): Subset of self.core_vars to enforce non-positivity
                - 'segmentation_net_dims' (list[int]): The numbers of hidden nodes for the segmentation NW layers.
                - 'segmentation_dropout_rate' (float): Dropout rate for the segmentation net.
                - 'weight_decay_segmentation' (float): Weight decay for the segmentation net.
                - 'weight_decay_embedding' (float): Weight decay for the embedding layers.
                - 'learning_rate' (float): Learning rate for the optimizer.

        Dimensions Guide:
            n_unique_chids: N, max(n_alts): J, batch_size (# of long input rows): B; NJ=B for ideal input data
            n_latent_classes: K, n_seg_vars: S, n_core_vars: C, n_emb_vars: E, n_sum_of_categories_across_emb_vars: Z
        """
        
        super().__init__()
        self.save_hyperparameters(config)

        # --- Store variable names and modes for easy access ---
        # The .get() method safely retrieves a value from the config dictionary.
        # The second argument is a default value to use if the key is not found.
        self.seg_vars = self.hparams.get("segmentation_vars", [])
        self.emb_vars = list(self.hparams.embedding_dims.keys()) # Order matters
        self.core_vars = self.hparams.get("core_vars", [])
        self.np_vars = self.hparams.get("non_positive_core_vars", [])
        self.embedding_mode = self.hparams.get("embedding_mode", "shared")
        self.choice_mode = self.hparams.get("choice_mode", "heterogeneous")

        # --- Calculate Embedding Offsets and Total Size ---
        # Create a tensor of offsets for indexing into the global embedding matrix: i.e., E separator
        if len(self.emb_vars)>0: # e.g., length E 1D tensor [0, 2, 6, 10, 12, 15, 18, 20]
            self.emb_offsets = torch.tensor([0] + list(self.hparams.embedding_dims.values())[:-1], dtype=torch.long).cumsum(dim=0)
        else:
            self.emb_offsets = torch.empty(0, dtype=torch.long)
        
        self.total_emb_categories = sum(self.hparams.embedding_dims.values()) #break down E to get Z

        # --- MODEL DEFINITION BASED ON NUMBER OF LATENT CLASSES ---
        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
            print(f"Initializing a Latent Class Model with K={self.hparams.n_latent_classes} classes.")

            # 1. Segmentation Component (Required for K > 1), with Dropout
            if len(self.seg_vars) == 0:
                raise ValueError(
                    "For a latent class model (n_latent_classes > 1), "
                    "'segmentation_vars' must be provided in the config."
                )
            dropout_rate = self.hparams.get("segmentation_dropout_rate", 0.0)
            print(f"Initializing segmentation network with dropout rate: {dropout_rate}")
            
            seg_layers = []
            layer_dims = self.hparams.segmentation_net_dims
            for i in range(len(layer_dims) - 1):
                seg_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                # Don't add ReLU or Dropout after the final layer that produces logits
                if i < len(layer_dims) - 2:
                    seg_layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        seg_layers.append(nn.Dropout(p=dropout_rate))
            self.segmentation_net = nn.Sequential(*seg_layers)
            # Add a safeguard to prevent saturation (ie deterministic assignment to a single class)
            if hasattr(self, "segmentation_net") and self.segmentation_net is not None:
                if hasattr(self.segmentation_net[-1], "bias") and self.segmentation_net[-1].bias is not None:
                    torch.nn.init.zeros_(self.segmentation_net[-1].bias)

            # 2. Global Embedding Layer(s) - CONDITIONAL ON MODE
            if self.embedding_mode == 'class-specific':
                print("Initializing with CLASS-SPECIFIC global embedding matrices.")
                # nn.Embedding module creates an (Z,J) learnable matrix expecting an index input [0,Z-1] and output z-th row with len J
                self.embedding_layers = nn.ModuleList([
                    nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives) # (Z,J)
                    for _ in range(self.hparams.n_latent_classes)
                ])
            else: # 'shared' mode
                self.embedding_layers = nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives)

            # 3. Coefficients (all have a class dimension)
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.core_vars))) # (K,C)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.emb_vars))) # (K,E)

            # 4. ASCs (have a class dimension)
            if self.choice_mode == 'heterogeneous':
                print("Initializing with Alternative Specific Constants (ASCs).")
                self.asc = nn.Parameter(torch.randn(self.hparams.n_latent_classes, self.hparams.n_alternatives - 1)) # (K,J-1)

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            print("Initializing a Single Class Model (K=1) without Segmentation Network.")

            # 1. Segmentation Component is SKIPPED

            # 2. Global Embedding Layer
            if len(self.emb_vars) > 0:
                self.embedding_layers = nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives) # (Z,J)

            # 3. Coefficients (no class dimension)
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(len(self.core_vars))) #(C,)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(len(self.emb_vars))) #(E,)

            # 4. ASCs (no class dimension)
            if self.choice_mode == 'heterogeneous':
                self.asc = nn.Parameter(torch.randn(self.hparams.n_alternatives - 1)) #(J-1,)
        
        try:
            # Embedding L2 Normalization ---
            self.project_embedding_rows_unit_l2()
        except Exception:
            pass  # Safe to skip; on_fit_start will handle it

        # non-positivity mask
        if self.np_vars:
            unknown = sorted(set(self.np_vars) - set(self.core_vars))
            if unknown:
                print(f"[DCM_SEAL] Ignoring unknown non_positive_core_vars: {unknown}, not defined in config['core_vars']")
            self._core_np_idx = [i for i, name in enumerate(self.core_vars) if name in set(self.np_vars)]
        else:
            self._core_np_idx = []

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        This method is aligned with a DataLoader that separates per-alternative data
        (e.g., core_features) from per-scenario data (e.g., seg_features, x_emb).
        Legend:
            B: batch size (#chids in batch)
            J: #alternatives (<= config["n_alternatives"])
            C: #core variables in this run
            E: #embedding variables (count, not total categories)
            Z: total categories across all embedding variables (sum of cardinalities)
            K: #latent classes
        Args:
            batch (dict): A dictionary from the DataLoader with efficient shapes:
                          'core_features': (B, J, C)
                          'seg_features':  (B, S)
                          'x_emb':         (B, E)
                          'mask':          (B, J)  - Boolean; 1 for real alternatives.
                          'choice':        (B,)
    
        Returns:
            torch.Tensor: A tensor of final utilities of shape (B, J).
        """
        # Get tensors and add offsets for individual x_embs (e.g., [0,3,2]) to be global incices ([0,5,9]) 
        x_core = batch['core_features']  # Shape: (B, J, C)
        x_seg = batch['seg_features']  # Shape: (B, S)
        x_emb_with_offsets = batch['x_emb'] + self.emb_offsets.to(self.device) # broadcast: (B, E) + (E,)

        # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
        if self.hparams.n_latent_classes > 1:
            # 1. --- Segmentation Probabilities ---
            class_logits = self.segmentation_net(x_seg)
            class_probs = F.softmax(class_logits, dim=1)  # Shape: (B, K)

            # 2. --- Core Utility (per-alternative) ---
            '''explanation of einsum: kc, bjc->bjk (RHS can be any combination using b j k but we deliberately do bjk)
            einsum: broadcasted dot products are done with vectors of the shared letter (dimension), for our case: c
            b0 k1 (C=3 J=2) example: [beta_k1c0, beta_k1c1, beta_k1c2] dot [[x_b0j0c0, x_b0j0c1, x_b0j0c2]:for alt j=0,
            [x_b0j1c0, x_b0j1c1, x_b0j1c2]:for alt j=1]-> [CoreUtil_j0, CoreUtil_j1] this happens for every b and k ->(B,J,K)
            '''
            utility_core_by_class = torch.zeros(1, device=self.device)
            if self.core_vars:
                core_betas = self.core_betas
                if self._core_np_idx: # Enforce ≤ 0 if any, by building a boolean mask
                    mask = torch.zeros(core_betas.size(1), dtype=torch.bool, device=core_betas.device)  # (C,) initialize as Falses
                    mask[self._core_np_idx] = True # mask nonpos True; below applies -F.softplus for True while keeping false
                    core_betas = torch.where(mask.unsqueeze(0), -F.softplus(core_betas), core_betas)    # (K, C)
                utility_core_by_class = torch.einsum('kc,bjc->bjk', core_betas, x_core) # note: self.core_betas is unconstrained

            # 3. --- Embedding Utility (per-alternative) ---
            '''
            Explanation of an element in raw_embs definition self.embedding_layers[k][name](x_emb[name]):
                layer=self.embedding_layers[k]: the k-th embedding layer (assoc. with LC k) of our module list defined above
                x_emb_with_offsets: A tensor with integer offsetted index for emb var's dim: (B,E), can be indexed to look like (B,E,Z)
                The object 'layer' itself is a function including learnable weight matrix that USES x_emb_with_offsets AS ITS INPUT.
                Ultimately, it is equivalent to do B times of input (E,Z) matmul by (Z,J) weight matrix to return (B,E,J)
            '''
            utility_from_embeddings_per_class = 0.0
            if self.emb_vars:  # E > 0
                pos_betas = F.softplus(self.embedding_betas)  # (K,E) if 'class-specific' or (E,) if 'shared'
                if pos_betas.ndim == 1:  # ie when self.embedding_mode == 'shared'
                    pos_betas = pos_betas.unsqueeze(0).expand(self.hparams.n_latent_classes, -1)  # (K,E)

                if self.embedding_mode == "class-specific": # Compute embedding per class because weights differ
                    class_utils = []
                    for k in range(self.hparams.n_latent_classes):
                        raw_embs = self.embedding_layers[k](x_emb_with_offsets) # see above explanation: (B,E,J)
                        norm_embs = raw_embs # Just for a formality; need to tweak this if you want to add embedding dropout
                        betas_k = pos_betas[k].unsqueeze(0).unsqueeze(2) # class-specific betas (1, E, 1)
                        util_k = torch.sum(norm_embs * betas_k, dim=1) # (B, E, J) * (1, E, 1), sum over E -> (B, J) 
                        class_utils.append(util_k)
                    utility_from_embeddings_per_class = torch.stack(class_utils, dim=-1) # (B, J) utils stacked K times -> (B,J,K)

                else: # ie 'shared' : apply embedding ONCE, then loop betas over K; dimensions are same
                    raw_embs = self.embedding_layers(x_emb_with_offsets)
                    norm_embs = raw_embs
                    class_utils = []
                    for k in range(self.hparams.n_latent_classes):
                        betas_k = pos_betas[k].unsqueeze(0).unsqueeze(2)
                        util_k = torch.sum(norm_embs * betas_k, dim=1)
                        class_utils.append(util_k)
                    utility_from_embeddings_per_class = torch.stack(class_utils, dim=-1) # (B,J,K)

            # 4. --- Combine Utilities and Add ASCs ---
            class_specific_utility = utility_core_by_class + utility_from_embeddings_per_class

            if self.choice_mode == 'heterogeneous':
                zeros = torch.zeros(self.hparams.n_latent_classes, 1, device=self.device) # (K, 1)
                full_asc = torch.cat([zeros,self.asc], dim=1) #concat (K, 1) and (K,J-1) -> (K, J)
                asc_reshaped = full_asc.T.unsqueeze(0) # (1,J,K)
                class_specific_utility += asc_reshaped # broadcasted sum: (B,J,K)

            # 5. --- Final Weighted Utility) ---
            final_utility = torch.sum(class_specific_utility * class_probs.unsqueeze(1), dim=2) # (B,J,K)* (B,1,K) dim k

        # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
        else:
            # 1. --- Core Utility (per-alternative) ---
            utility_core = torch.zeros(1, device=self.device)
            if self.core_vars:
                core_betas = self.core_betas
                if self._core_np_idx:
                    mask = torch.zeros(core_betas.size(0), dtype=torch.bool, device=core_betas.device)  # (C,)
                    mask[self._core_np_idx] = True
                    core_betas = torch.where(mask, -F.softplus(core_betas), core_betas)                 # (C,)
                utility_core = torch.einsum('c,bjc->bj', core_betas, x_core) # note: self.core_betas is unconstrained

            # 2. --- Embedding Utility (per-alternative) ---
            utility_emb = torch.zeros(1, device=self.device)
            if self.emb_vars:
                positive_betas = F.softplus(self.embedding_betas) # Shape: (E,)
                raw_embs = self.embedding_layers(x_emb_with_offsets) # Shape: (B, E, J)
                norm_embs = raw_embs  # Just for a formality; need to tweak this if you want to add embedding dropout
                utility_emb = torch.sum(norm_embs * positive_betas.unsqueeze(0).unsqueeze(2), dim=1)

            # 3. --- ASC Utility (per-alternative) ---
            utility_asc = torch.zeros(1, device=self.device)
            if self.choice_mode == 'heterogeneous':
                zeros = torch.zeros(1, device=self.device) #just a scalar 0
                full_asc = torch.cat([zeros,self.asc.reshape(-1)],dim=0) # ensure 1D; reshape: infer this dimension
                utility_asc = full_asc.unsqueeze(0)

            # 4. --- Final Utility ---
            final_utility = utility_core + utility_emb + utility_asc # no unsqueeze is needed for K=1

        # --- Final Step: Apply Alternative Masking ---
        final_utility = final_utility.masked_fill(~batch['mask'], -torch.inf)
        return final_utility

    def project_embedding_rows_unit_l2(self):
        """
        Enforce row-wise unit L2 norm on the embedding weight matrix/matrices.

        Shapes:
          - Shared embedding:           W {in} R^(Z, J)
          - Class-specific embeddings:  {W_k} with each W_k {in} R^(Z, J), k=1..K
        We normalize along dim=1 (rows), so ||W[z, :]||2 = 1 for all z.
        """
        # If no embedding variables in this run, nothing to do
        if len(self.emb_vars) == 0:
            return

        with torch.no_grad():
            if self.embedding_mode == "class-specific" and self.hparams.n_latent_classes > 1:
                for layer in self.embedding_layers:          # layer: nn.Embedding(Z, J)
                    W = layer.weight                          # (Z, J)
                    norms = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    W.div_(norms)                             # in-place row-wise normalize
            else:
                W = self.embedding_layers.weight              # (Z, J)
                norms = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                W.div_(norms)

    # Lightning executes the below five methods without explicit calls (i.e., reserved names or "Lightning Hooks")
    # The first two are substituting the normalization for embedding weight matrix, which was slow in legacy
    
    # one-time normalize at the start
    def on_fit_start(self):
        self.project_embedding_rows_unit_l2()

    # project after each optimizer step
    def optimizer_step(self, *args, **kwargs):
        # run the default step first
        super().optimizer_step(*args, **kwargs)
        # then project rows back to unit L2
        self.project_embedding_rows_unit_l2()
        
    def on_train_epoch_start(self):
        self._tr_ll_sum  = torch.tensor(0.0, device=self.device)
        self._tr_ll0_sum = torch.tensor(0.0, device=self.device)

    def on_validation_epoch_start(self):
        self._va_ll_sum  = torch.tensor(0.0, device=self.device)
        self._va_ll0_sum = torch.tensor(0.0, device=self.device)

    def on_test_epoch_start(self):
        self._te_ll_sum  = torch.tensor(0.0, device=self.device)
        self._te_ll0_sum = torch.tensor(0.0, device=self.device)

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer with different weight_decay values for
        different parameter groups.
        """
        # Get the regularization hyperparameters, with defaults
        decay_seg = self.hparams.get("weight_decay_segmentation", 1e-2) # Default: Strong decay
        decay_emb = self.hparams.get("weight_decay_embedding", 1e-4) # Default: Moderate decay

        # --- Create Parameter Groups ---
        # Group 1: The segmentation network parameters
        # This group only exists for the latent class model (K > 1)
        param_groups = []
        if self.hparams.n_latent_classes > 1:
            param_groups.append({
                'params': self.segmentation_net.parameters(),
                'weight_decay': decay_seg
            })

        # Group 2: The embedding layer weights
        if len(self.emb_vars) > 0:
            param_groups.append({
                'params': self.embedding_layers.parameters(),
                'weight_decay': decay_emb
            })

        # Group 3: All other betas and ASCs, with NO weight decay.
        # We collect them in a list first to handle cases where they might not exist.
        no_decay_params = []
        if len(self.core_vars) > 0:
            no_decay_params.append(self.core_betas)
        if len(self.emb_vars) > 0:
            no_decay_params.append(self.embedding_betas)
        if self.choice_mode == 'heterogeneous' and hasattr(self, 'asc'):
            no_decay_params.append(self.asc)

        if no_decay_params: # Only add the group if it's not empty
            param_groups.append({'params': no_decay_params,'weight_decay': 0.0})

        # --- Initialize the Optimizer ---
        optimizer = torch.optim.AdamW(param_groups,lr=self.hparams.learning_rate)
        return optimizer

    def _calculate_loss(self, batch):
        """
        Helper function to calculate loss.
        """
        # 1. Get Utilities from the forward pass
        # The shape of utilities is now (BatchSize, NumAlternatives)
        utilities = self.forward(batch)

        # 2. Get the index of the chosen alternative for each scenario in the batch
        # This comes directly from the dataloader.
        choice_indices = batch['choice'] # Shape: (BatchSize,) indices of chosen alt (0 to J-1)

        # 3. Calculate cross-entropy loss
        # This is the standard negative log-likelihood for choice models.
        # It implicitly applies a softmax to the utilities.
        loss = F.cross_entropy(utilities, choice_indices)
        # --- Anti-saturation on membership (K>1 only) ---
        K = self.hparams.n_latent_classes
        if K > 1: 
            p = F.softmax(self.segmentation_net(batch['seg_features']), dim=1) # (B, K)
            entropy = -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()/np.log(K) # uncertainty: (0:deterministic to 1 uniform)
            lam = float(self.hparams.get("seg_entropy_weight", 0)) # incentivize uncertainty (via - sign)
            if lam>0:
                loss = loss + lam * (entropy-0.5)**2 # This is still not perfect... not enforced in the paper
            self.log("scaledentropy", entropy, on_epoch=True, prog_bar=False)

        # Calculate accuracy
        preds = torch.argmax(utilities, dim=1)
        acc = (preds == choice_indices).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        # accumulate LL and LL0 for rho^2
        loss, acc = self._calculate_loss(batch)
        self._tr_ll_sum  += _ll_from_utilities(utilities=self.forward(batch),
                                       y_idx=batch['choice'],
                                       row_mask=None)  # mask not needed; forward already masked
        self._tr_ll0_sum += _ll0_uniform(mask=batch.get('mask', None),
                                         batch_size=batch['choice'].size(0),
                                         n_alts=self.hparams.n_alternatives,
                                         device=self.device)
        # Log metrics for the epoch.
        self.log_dict({'train_loss': loss, 'train_acc': acc},
                      on_step=False, # We only need the final epoch value
                      on_epoch=True,
                      prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        loss, acc = self._calculate_loss(batch)
        self._va_ll_sum  += _ll_from_utilities(self.forward(batch), batch['choice'], None)
        self._va_ll0_sum += _ll0_uniform(batch.get('mask', None),
                                         batch['choice'].size(0),
                                         self.hparams.n_alternatives,
                                         self.device)
        # The final, correct values will be logged at the end of the epoch.
        self.log_dict(
            {'val_loss': loss, 'val_acc': acc},
            on_epoch=True,
            prog_bar=True) # It's often useful to see val_loss in the bar

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        """
        loss, acc = self._calculate_loss(batch)
        self._te_ll_sum  += _ll_from_utilities(self.forward(batch), batch['choice'], None)
        self._te_ll0_sum += _ll0_uniform(batch.get('mask', None),
                                         batch['choice'].size(0),
                                         self.hparams.n_alternatives,
                                         self.device)
        self.log_dict({'test_loss': loss, 'test_acc': acc},
                      on_epoch=True,
                      prog_bar=True)

    #other lightning hooks for rhosq
    def on_train_epoch_end(self):
        self.log("train/rho2", _rho2(self._tr_ll_sum, self._tr_ll0_sum), prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log("val/rho2", _rho2(self._va_ll_sum, self._va_ll0_sum), prog_bar=True)
    
    def on_test_epoch_end(self):
        print(f'Test data LL0: {self._te_ll0_sum}, LLB: {self._te_ll_sum}')
        self.log("test/rho2", _rho2(self._te_ll_sum, self._te_ll0_sum), prog_bar=False)

    @property
    def beta_raw(self):
        """
        Unconstrained parameters, concatenated as [core | embedding].
        Rows are expanded to K where necessary for shape compatibility.
        """
        def to_2d(t):
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            return t.unsqueeze(0) if t.ndim == 1 else t  # (D,) -> (1,D)

        K = self.hparams.n_latent_classes

        # Core block
        if len(self.core_vars) > 0:
            b_core = to_2d(self.core_betas)                  # (K or 1, C)
            if b_core.size(0) == 1 and K > 1: # if latent class but "shared"
                b_core = b_core.expand(K, -1)                # (K, C) 
        else:
            b_core = torch.zeros((K, 0), device=self.device) # (K, 0)

        # Embedding block
        if len(self.emb_vars) > 0:
            b_emb = to_2d(self.embedding_betas)              # (K or 1, E)
            if b_emb.size(0) == 1 and K > 1:
                b_emb = b_emb.expand(K, -1)                  # (K, E) for 'shared'
        else:
            b_emb = torch.zeros((K, 0), device=self.device)  # (K, 0)

        return torch.cat([b_core, b_emb], dim=-1)            # (K, C+E)

    @property
    def beta(self):
        """
        Constrained parameters for reporting, matching forward():
          - embedding betas:   ≥ 0 via softplus
          - selected core betas (non_positive_core_vars): ≤ 0 via -softplus
          - other core betas:  unconstrained
        Concatenated as [core | embedding], shape (K, C+E).
        """
        K = self.hparams.n_latent_classes

        # --- Core (apply ≤0 only to requested names) ---
        if len(self.core_vars) > 0:
            b_core = self.core_betas
            if b_core.ndim == 1:
                b_core = b_core.unsqueeze(0)                 # (1, C)
            if b_core.size(0) == 1 and K > 1:
                b_core = b_core.expand(K, -1)                # (K, C)

            if getattr(self, "_core_np_idx", None):
                mask = torch.zeros(b_core.size(1), dtype=torch.bool, device=b_core.device)  # (C,)
                mask[self._core_np_idx] = True
                b_core = torch.where(mask.unsqueeze(0), -F.softplus(b_core), b_core)        # (K, C)
        else:
            b_core = torch.zeros((K, 0), device=self.device)

        # --- Embedding (always ≥0) ---
        if len(self.emb_vars) > 0:
            b_emb = self.embedding_betas
            if b_emb.ndim == 1:
                b_emb = b_emb.unsqueeze(0)                   # (1, E)
            if b_emb.size(0) == 1 and K > 1:
                b_emb = b_emb.expand(K, -1)                  # (K, E) for 'shared'
            b_emb = F.softplus(b_emb)                        # (K, E)  non-negative
        else:
            b_emb = torch.zeros((K, 0), device=self.device)

        return torch.cat([b_core, b_emb], dim=-1)            # (K, C+E)

    def get_embedding_weights(self,inds,cols, conv_file_path: str = None):
        """
        Extracts the full, trained embedding weight matrix (Z, J) from the
        embedding layer(s).
    
        Args:
            conv_file_path (str, optional): Path to the dfConv.csv file. If provided,
                the output DataFrame will have a detailed MultiIndex with original
                category labels for the rows. Defaults to None.
    
        Returns:
            pandas.DataFrame or dict:
            - If 'shared' mode: A single DataFrame of shape (Z, J).
            - If 'class-specific' mode: A dictionary of DataFrames, one for each class k.
        """

        # Set model to evaluation mode and disable gradients
        self.eval()
        with torch.no_grad():
            if self.hparams.n_latent_classes > 1 and self.embedding_mode == 'class-specific':
                # --- PATH 1: CLASS-SPECIFIC MODE ---
                class_weights = {}
                for k in range(self.hparams.n_latent_classes):
                    # Access the .weight attribute of the embedding layer for class k
                    weight_matrix = self.embedding_layers[k].weight.cpu().numpy()
                    class_weights[k] = pd.DataFrame(weight_matrix, index=inds, columns=cols)
                return class_weights
    
            else:
                # --- PATH 2: SHARED or K=1 MODE ---
                # Access the .weight attribute of the single embedding layer
                weight_matrix = self.embedding_layers.weight.cpu().numpy()
                return pd.DataFrame(weight_matrix, index=inds, columns=cols)
# DCM-SEAL has been configured.

if __name__ == '__main__':
    _=OrderedDict() #to suppress import warning
