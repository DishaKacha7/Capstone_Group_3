# Temporal Patterns and Pixel Precision: Satellite-Based Crop Classification Using Deep Learning and Machine Learning

## Project Overview

This repository contains the complete code and models developed for **crop type classification** using Sentinel-2 remote sensing imagery. We perform:

- **Pixel-Level and Field-Level classification** using classical machine learning models (XGBoost, LightGBM, Random Forest, Logistic Regression).
- **Pixel-Level and Field-Level classification** using deep learning models (1D CNN, TabTransformer, CNN + BiLSTM + Focal Loss).
- **Patch-Level classification** using deep learning models (Multi-Channel CNN, 3D CNN, Transformer-Based model, and Ensemble Architectures).

The project systematically benchmarks different modeling strategies for high-resolution crop type mapping.

## Data Access

All datasets are hosted on Google Drive. Use the following commands to download them easily with `gdown`:

```bash
# For Classical ML Models
pip install gdown
gdown https://drive.google.com/uc?id=1tywwZdycgBTKgkNAZMKyCL5skaezzE3O -O final_data.parquet

# For Deep Learning Pixel level Models
gdown https://drive.google.com/uc?id=1QFCA6AIF85MtbO4oDtA_64I8GwjEZJd8 -O merged_dl_258_259.parquet

# For Deep Learning Patch Level Models
Run Create_Patches.py and Create Master Data to generate the parquet, which will be used in all subsequent code files

```


> **Note:** Please ensure you downlod them into the project root directory and in a folder called Data.


## Repository Structure

```bash
Capstone_Group_3-main/
├── README.md
├── Presentation/
│   └── Capstone Plan.pptx
├── src/
│
│   ├── Classical Machine Learning/
│   │   ├── Field Level/
│   │   │   ├── EDA Field Level.py
│   │   │   ├── Ensemble - Voting and Stacking.py
│   │   │   ├── SMOTE_meta.py
│   │   │   ├── inference_classical_ensemble.py
│   │   │   ├── xg_boost_hyper.py
│   │   └── pixel_level/
│   │       ├── base_ml_models.py
│   │       ├── pixel_voting.py
│
│   ├── Deep Learning/
│   │   ├── Patch Level/
│   │   │   ├── 3D_CNN.py
│   │   │   ├── Create Master Data.py
│   │   │   ├── Create_Patches.py
│   │   │   ├── Ensemble - 3D CNN.py
│   │   │   ├── Inference_Ensemble.py
│   │   │   ├── Multi_Channel_CNN.py
│   │   │   ├── results_3d_cnn.py
│   │   │   ├── results_ensemble_patching.py
│   │   │   ├── results_multi_channel_cnn.py
│   │   │   ├── results_transformer_patching.py
│   │   └── Pixel_Field_Level/
│   │       ├── TabTransformer.py
│   │       ├── TabTransformer_Final_Field.py
│   │       ├── best_ccn_params.py
│   │       ├── cnn_bilstm.py
│   │       ├── cnn_dl_hyper.py
│   │       ├── field_acc_cnnlstm.py

```

## Key Models

### Classical ML Models (Field-Level)
- **XGBoost** with extensive Optuna hyperparameter tuning. ```xg_boost_hyper.py```
- **LightGBM** and **Random Forest** with field-level aggregation. 
- **Voting and Stacking** ensemble classifiers. ```Ensemble - Voting and Stacking.py```

### Deep Learning Models (Pixel-Level and Field-Level)
- **1D CNN** trained with Optuna for hyperparameter tuning. ```cnn_dl_hyper.py```
- **Ensemble TabTransformer** for handling structured tabular data. ```TabTransformer.py```
- **CNN + BiLSTM with Focal Loss** to handle temporal patterns and imbalance. ```cnn_bilstm.py```

### Patch-Level Models
- **Multi-Channel 2D CNN** for spatial texture learning.
- **3D CNN** for spectral-temporal feature extraction.
- **Transformer-Based Model** for attention-based patch learning.
- **Ensemble Architecture** combining all patch-level models for robust predictions.

> **Note:** The files mentioned above under Key Models are all used for modelling, to get inferences on the data and evaluations there are different files.

## Results Summary

| Approach         | Kappa | F1   | Key Highlights                                      |
|------------------|-------|------|-----------------------------------------------------|
| Classical ML     | ~0.69 | ~0.77| Voting Ensemble highest among classical models      |
| Deep Learning    | ~0.77 | ~0.84| CNN + BiLSTM + Focal Loss strongest performer        |
| Patch-Level DL   | ~0.66 | ~0.75| Ensemble of CNN + Transformer best patch-level model |


## Getting Started

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run any model of your choice. Example:

```bash
python classical_ml/xg_boost_hyper.py
python deep_learning/cnn_bilstm.py
```
This trains the model and generates the weights to be used in inferencing

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to fork, improve, and create pull requests!

---

### Maintainer

- Devarsh Apurva Sheth, Disha Kacha, Sairam Venkatachalam (George Washington University)
- Data Science Graduate Student
- Passionate about Deep Learning and Remote Sensing 🚀

---

> *"Science is the poetry of reality."* – Richard Dawkins
