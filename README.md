# STMT: Time Information-enhanced Multi-space Modeling Spatiotemporal Transformer Network for Traffic Flow Forecasting

## Overview

This repository contains the implementation of **STMT** 

## 📢 Status

**Note:** The full model code will be released upon paper acceptance.

Currently available:
- Configuration files for different datasets
- Training utilities and data preprocessing scripts
- Training logs in the `study/` folder

## 📁 Repository Structure

```
STMT/
├── config/           # Configuration files for different datasets
│   ├── pems03.json
│   ├── pems04.json
│   ├── pems07.json
│   ├── pems08.json
│   ├── jinan.json
│   └── sd.json
├── study/            # Training logs
│   └── logs/
├── engine.py         # Training engine
├── train.py          # Training script
├── util.py           # Utility functions
└── generate_training_data.py  # Data preprocessing
```

## 📊 Datasets

Our experiments are conducted on multiple real-world traffic datasets:

- **PEMS03/04/07/08**: Traffic data from California (PeMS)
  - Source: [STSGCN Repository](https://github.com/Davidham3/STSGCN.git)
- **SD**: Traffic data from San Diego
  - Source: [LargeST Repository](https://github.com/liuxu77/LargeST.git)
- **Jinan**: Traffic data from Jinan, China
  - Source: [STDN Repository](https://github.com/roarer008/STDN.git)

### Data Download

Preprocessed datasets are available for download:

📦 [Download Processed Datasets](https://drive.google.com/file/d/1dq-35bFMltJ_Jh8G2rJQiR27SJEVLJeG/view?usp=sharing)

After downloading, extract the data into the `data/` directory.

## 🚀 Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.1.1
- CUDA 12.1 (for GPU support)

```bash
pip install -r requirements.txt
```

### Running the Code

To train the model on a specific dataset:

```bash
python -m train --dataset pems08
```

Available datasets: `pems03`, `pems04`, `pems07`, `pems08`, `jinan`, `sd`

## 🔬 Training Logs

The `study/logs/` directory contains detailed training logs for all experiments across different datasets.

