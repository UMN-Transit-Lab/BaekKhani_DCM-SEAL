# DCM-SEAL

**Discrete Choice Model with Segmentation & Embedding via Adaptive Learning**

This repository accompanies the academic paper **"A Discrete Choice Model with Segmentation and Embedding via AI-Learning (DCM-SEAL)"** (`BaekKhani2026_DCMSEAL.pdf`). The code operationalizes the modeling framework described in the paper by combining **latent-class segmentation, class-specific embeddings, and interpretable linear-in-parameter utilities** within a PyTorch Lightning workflow.

---

## Features

- **Flexible latent-class architecture** – jointly trains segmentation networks, class-specific embeddings, and alternative-specific constants to capture behavioral heterogeneity.
- **Efficient dataset handling** – includes a pre-padded `PaddedChoiceDataset` for O(1) item retrieval and drop-in compatibility with standard PyTorch DataLoader collate functions.
- **Automated data preparation** – streamlines categorical encodings, segmentation variables, and train/test splits.
- **End-to-end training runner** – loads data, constructs `DataLoader` instances, and trains/evaluates the model via PyTorch Lightning.
- **Hyperparameter optimization** – leverages Optuna with adaptive batch-size and epoch scheduling.

---

## Setup Requirements

The project is tested with **Anaconda Python 3.12**. To reproduce the environment:

1. Create an environment, install basic dependencies, then activate:
   ```bash
   conda create -n dcmseal python=3.12 numpy pandas scikit-learn optuna
   conda activate dcmseal
   ```
2. Install ML dependencies (See PyTorch official page to tweak it for a GPU-CUDA enabled machine):
   ```bash
   conda install -c conda-forge pytorch lightning optuna-integration tensorboard
   ```


---

## Repository Structure

```text
.
├── BaekKhani2025_DCMSEAL.pdf       # Full academic manuscript
├── README.md                       # Project overview & instructions
├── data/Synthesized                # Input dataset: Synthesized
│   ├── genSynthData.py             # Synthetic data generator (Appendix A)
│   └── trueMRS.csv                 # For Synthetic data MRAE calculation and class reordering
├── data/SwissMetro                 # Input dataset: Synthesized
│   ├── preprocessSM.py             # Preprocessing datawide.csv (the 9,036 full observations)
│   └── filteredSwissMetro.csv      # SwissMetro observations with full info (some column names changed)
├── mlogit/mlogit.R                 # Multinomial logit benchmarks
├── results/                        
│   ├── 251020_Synthetic/           # Important synthetic experiment outputs & logs (Section 4)
│   └── 251026_SwissMetroFinal/     # Important SwissMetro experiment outputs & logs (Section 5)
├── src/                            
│   ├── data_loader.py              # Dataset classes & pre-padded dataset implementation
│   ├── data_processing.py          # CSV loading & preprocessing utilities
│   ├── dcm_seal_model.py           # Core PyTorch Lightning model definition
│   ├── reporting_estimates.py      # For model result interpretation
│   └── run_model.py                # Stand-alone training entry point
└── main.py                         # Optuna hyperparameter search driver
```

---

## Getting Started

### 1) Prepare data

Place preprocessed `dfIn.csv` (and optionally `dfConv.csv` for categorical conversions) inside a dataset-specific directory:

```
data/<DatasetName>/
```

### 2) Run a single experiment

Edit the configuration dictionary in `src/run_model.py`, or import and pass a configuration from the first block of `main.py`:

```bash
python src/run_model.py
```

### 3) Hyperparameter search

Set `DATA2USE` in `main.py`, then launch Optuna-driven experiments:

```bash
python main.py
```

When finished, it will raise the exception 'exploration has been completed; proceed with an IDE env for post-model analyses'
Artifacts are saved to `logs/` for training metrics and `checkpoints/` for top-performing models.

### 4) Final model selection and post-model analyses

Various codes for the paper are stored in the `#%% Post-Search Codes` block, which requires to run the first `#%% Initial Settings` block.
It is currently designed to run on an IDE (e.g., VB Studio, Spyder, Jupyter) and requires dynamic user inputs.

---



## Contact

**Kwangho Baek**  
baek0040@umn.edu · dptm22203@gmail.com
