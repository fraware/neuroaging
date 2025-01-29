# Neuro-Aging

This repository provides a pipeline for predicting human brain age from EEG data using both the **Healthy Brain Network (HBN)** dataset and the **Brain Age Challenge** dataset. It includes:

- Scripts to **download** and **prepare** the HBN data.  
- A **preprocessing** pipeline for filtering, downsampling, bad-channel detection, and power spectral density (PSD) extraction.  
- A **training** script to learn an SVR model (with optional nested cross-validation).  
- A **prediction** script to generate predictions on new (test/validation) EEG data.

## Repository Structure

```plaintext
neuroaging/
├── .gitignore
├── LICENSE
├── README.md               <-- You are here
├── requirements.txt        <-- Python dependencies
├── download_data.py        <-- Download HBN participants file & .mat/.npy data
├── preprocessing.py        <-- EEG preprocessing & PSD computation scripts
├── train.py                <-- Script to load preprocessed data & train the SVR
└── predict.py              <-- Script to load a saved model & predict on new data
```

## Installation

1. **Clone or download** this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/neuroaging.git
   cd neuroaging

2. **Install the required Python libraries**:
   
   ```bash
   pip install -r requirements.txt

## Data

### 1. Healthy Brain Network (HBN)
- The HBN dataset offers resting-state EEG recordings in `.mat` format.  
- `download_data.py` automates downloading the **participants file** and `.mat` files from the HBN S3 bucket.  
- By default, it saves outputs under `data/HBN_raw` (but you can edit paths in the script).

### 2. Brain Age Challenge Data
- We assume you have the Brain Age Challenge `.fif` EEG files already (due to competition constraints).  
- Place them under, for example, `data/NeuroTex/raw_train_tutorial`, or adjust any references in `preprocessing.py` and `train.py`.

> **Important:** Large `.npy`, `.mat`, or `.fif` files are typically **not** included in this repository.  
> You must place or download them before running.

## Usage

Below is the recommended workflow. Note: Customize directory paths inside the Python scripts (e.g., `raw_dir`, `preprocessed_dir`, `psd_dir`) to match your local setup.

1. **Download HBN data** (Optional if you already have it):

   ```bash
   python download_data.py

This will fetch `participants.tsv` and each subject’s `.mat` file from the S3 bucket, converting them to `.npy`.

2. **Preprocess** EEG data to filter, downsample, detect bad channels, split EC/EO, and compute PSD:

   ```bash
   python preprocessing.py

This script covers HBN data by default (`preprocess_hbn_data` and `compute_psd_hbn`).
For Brain Age `.fif` files, you should create or adapt similar functions (e.g., `preprocess_brain_age_data`), also in `preprocessing.py`.

3. **Train the model**:

   ```bash
   python train.py

Loads PSD features from both HBN and Brain Age directories, combines them, and trains an SVR model.
The script saves the trained model as `svr_brain_age_model.joblib`.

4. **Predict on new data**:

   ```bash
   python predict.py

Loads the saved `.joblib` model and uses it to predict the ages for unseen/test EEG data.
Outputs a `submission.csv` file that can be used for challenge submissions.

## Customizing the Pipeline

- **File Paths**
All scripts contain variables at the top specifying input/output directories (e.g., `HBN_OUTPUT_DIR`, `preproc_dir`, `psd_dir`). **Update these** to reflect your directory structure.

- **Hyperparameters**
By default, we use an SVR with a certain `C` and `kernel` (`train.py`). You can enable **nested cross-validation** (`nested_cv=True`) if you want automatic hyperparameter search, though it can be time-consuming.

- **Bad Channel Threshold**
In `preprocessing.py`, we skip recordings with **>30 noisy channels**. **Adjust as needed** if you want a different threshold.

## Troubleshooting

- **Permissions or Path Errors**: Make sure the directories used for data and output actually exist, or let the scripts create them automatically (`os.makedirs(..., exist_ok=True)`).
- **Large Files**: If you run out of space or memory, consider removing older intermediates or storing them on external drives.
- **Missing Dependencies**: If you see `ModuleNotFoundError`, verify you installed `requirements.txt` correctly.

## Contributing
Contributions are welcome! Feel free to open an issue or a pull request if you have fixes or enhancements.

## License
This project is licensed under the terms of the **[MIT License](https://opensource.org/license/mit)**. Refer to the [`LICENSE`](LICENSE) file for details.
