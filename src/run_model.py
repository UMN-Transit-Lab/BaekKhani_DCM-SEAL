# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import os
from pathlib import Path

if __name__ == "__main__":
    if Path.cwd().name=='src': # optional control for easier debugging
        os.chdir("..")

import warnings
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores for performance boost
torch.backends.cudnn.benchmark = True         # Let cuDNN pick fastest kernels for fixed shapes


# Import custom modules from the 'src' directory
from src.data_processing import load_and_preprocess_data
from src.data_loader import PaddedChoiceDataset
from src.dcm_seal_model import DCM_SEAL

def run_model(config:dict,data2use:str='Synthesized',verbose:bool=False,logname:str='experiment'):
    """Main function to run the DCM-SEAL model experiment."""

    # --- 1. Load and Preprocess Data, and Update Data-Driven Variable Options to Config ---
    print("--- Starting Data Preprocessing ---")
    data_dir = Path.cwd() / "data" / data2use
    train_df, test_df, train_x_emb, test_x_emb, discovered_embedding_dict = load_and_preprocess_data(
        config=config,data_dir=data_dir)

    # Add the discovered embedding dimensions to the main config.
    # This is crucial for initializing the model with the correct layer sizes.
    config["embedding_dims"] = discovered_embedding_dict['dims']
    config["embedding_labels"] = discovered_embedding_dict['labels']

    # Update the segmentation network input dimension based on the processed data
    # This makes the config robust to the number of one-hot columns created
    seg_vars_one_hot = [col for col in train_df.columns if any(f"{s}_" in col for s in config["segmentation_vars_categorical"])]
    seg_vars_cont = config.get("segmentation_vars_continuous", [])
    if seg_vars_one_hot:
        config["segmentation_vars"] = seg_vars_one_hot + seg_vars_cont
    else:
        config["segmentation_vars"] = seg_vars_cont
    total_seg_vars = len(seg_vars_one_hot) + len(seg_vars_cont)
    hidden_dims = config.get("segmentation_hidden_dims", []) # e.g., [32, 16]
    config["segmentation_net_dims"] = [total_seg_vars] + hidden_dims + [config["n_latent_classes"]]
    print(f"Updated segmentation network dimensions to: {config['segmentation_net_dims']}")

    # Recreate input dataframe
    dfInput=pd.concat([train_df,test_df])
    #dfInput[['chid','pid']].iloc[1::config['n_alternatives']].to_csv('sort.csv',index=False)
    dfInput=dfInput.loc[:,config['core_vars']+config['segmentation_vars']].reset_index(drop=True)
    dfEmb=pd.DataFrame(torch.cat([train_x_emb,test_x_emb],dim=0).numpy(),columns=list(config['embedding_dims'].keys()))
    dfInput=pd.concat([dfInput,dfEmb],axis=1)

    # --- 2. Create PyTorch Datasets and DataLoaders ---
    print("\n--- Creating Datasets and DataLoaders ---")
    fast_dataloader = config.get("dataloader_fast_mode", True)
    train_dataset = PaddedChoiceDataset(train_df, train_x_emb, config, fast_mode=fast_dataloader)
    test_dataset = PaddedChoiceDataset(test_df, test_x_emb, config, fast_mode=fast_dataloader)

    train_collate = train_dataset.collate_indices if fast_dataloader else None
    test_collate = test_dataset.collate_indices if fast_dataloader else None

    numwork = config.get("dataloader_num_workers", 0)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=numwork,
        pin_memory=True,
        collate_fn=train_collate
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=numwork,
        pin_memory=True,
        collate_fn=test_collate
        )

    # --- 3. Initialize and Train the Model ---
    print("\n--- Initializing Model and Trainer ---")
    model = DCM_SEAL(config)

    # Configure a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',      # Watch the validation loss every validation epoch
        dirpath='checkpoints/',
        filename= data2use +'_'+ logname+'_{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,            # Keep only the single best checkpoint
        mode='min'               # "Best" means the *lowest* val_loss
    )
    
    early_stop = EarlyStopping( #for Optuna speedup
        monitor="val_loss",    # Use validation loss as the stopping signal
        mode="min",            # We want it to go *down*
        patience=3,            # If it hasn’t improved for 3 validations, stop
        min_delta=0.0,         # Improvement must be at least this much
        check_finite=True,     # Stop if loss becomes NaN/Inf
    )
    
    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if use_gpu else "cpu", #"auto",
        devices=1, # don't recommend multi-gpu run for a single session; unstable
        precision="16-mixed" if use_gpu else 32,
        callbacks=[checkpoint_callback, early_stop], # Use the two callbacks above
        logger=pl.loggers.TensorBoardLogger("logs/", name=data2use + '_' + logname),
        log_every_n_steps=1000,
        enable_model_summary=False, # Skip the parameter summary to reduce overhead
        gradient_clip_val = 1 # Clip gradients at L2-norm to prevent spikes
)

    print("\n--- Starting Model Training ---")
    if verbose:
        batch = next(iter(train_loader))
        print(
            batch['core_features'].shape,  # (B, J_max, C)
            batch['mask'].shape,           # (B, J_max)
            batch['seg_features'].shape,   # (B, S)
            batch['x_emb'].shape,          # (B, E)
            batch['choice'].shape          # (B,)
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    final_metrics = {}
    for k, v in trainer.callback_metrics.items():
        try: # tensors -> float; others pass-through if numeric
            final_metrics[k] = float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
        except Exception: # ignore non-numeric entries
            pass 

    best_score = checkpoint_callback.best_model_score.item()
    return model, best_score, final_metrics, dfInput

#%% Smoke Run; first define DATA2USE and CONFIG in the first block of main.py (F9)
if __name__ != "__main__":
    DATA2USE=0
    CONFIG=0
else:
    data2use=DATA2USE
    config=CONFIG.copy()
    match data2use:
        case 'Synthesized':
            config.update({
                "non_positive_core_vars":['x1','x2','x3'],
                "n_latent_classes": 3,
                "embedding_mode": "class-specific", 
                "segmentation_hidden_dims": [128, 256, 128],
                "learning_rate": 0.002,
                "segmentation_dropout_rate": 0.2,
                "weight_decay_segmentation": 1e-2,
                "weight_decay_embedding": 1e-3,
                "batch_size": 512,
                "max_epochs": 50,
            })
        case 'SwissMetro':
            config.update({
                "non_positive_core_vars":["CO","TT"],
                "n_latent_classes": 2,
                "embedding_mode": "class-specific", 
                "segmentation_hidden_dims": [32,64,32],
                "learning_rate": 0.01,
                "segmentation_dropout_rate": 0.2,
                "weight_decay_segmentation": 0.02,
                "weight_decay_embedding": 0.02,
                "batch_size": 1024,
                "max_epochs": 10,
            })

    trained_model, bestobj, scores, dfInput = run_model(config,data2use,True)
    print(scores)
    from src.reporting_estimates import extract_betas, extract_embedding, predict_membership
    beta, beta_raw = extract_betas(trained_model, config)
    emb_df = extract_embedding(trained_model,config)
    membership_df=predict_membership(trained_model,dfInput)
    print(beta)
    print(beta_raw)
    print(emb_df)

