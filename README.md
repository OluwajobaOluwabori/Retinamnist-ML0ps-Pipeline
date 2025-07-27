# RetinaMNIST MLOps Project

This repository contains an **end-to-end MLOps pipeline** for the [RetinaMNIST dataset](https://medmnist.com/) (from MedMNIST v2). The project was developed as part of the **[DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)** final project.

## 🚀 Project Overview

The goal is to build a robust ML pipeline for **diabetic retinopathy severity prediction (ordinal regression)** using the RetinaMNIST dataset (3x224x224 retina fundus images).  
Key components include:

- **Data preparation** using MedMNIST.
- **Model training** with a ResNet-50 architecture.
- **Experiment tracking** and **model registry** using MLflow.
- **Pipeline orchestration** with Prefect.
- **Model deployment** using Flask + Docker.
- **Monitoring** with Evidently (data & prediction drift detection).
- **CI/CD** with best practices (tests, pre-commit hooks, Makefile, linter).

- CI/CD and MLOps best practices

## 📂 Project Structure

│
├── data/ # Raw and processed data
├── models/ # Saved models (ResNet-50 .pth)
├── app.py # Flask API for serving predictions
├── train_pipeline.py # Model training pipeline with MLflow
├── monitoring_flow.py # Prefect flow for model/data monitoring
├── model_utils.py # Utilities for MLflow and model loading
├── Dockerfile # Container setup for Flask API
├── Makefile # Build, test, and deployment commands
├── tests/ # Unit & integration tests
│ ├── test_model.py
│ └── test_app.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🛠 Tech Stack

- **ML Framework**: PyTorch
- **Experiment Tracking**: MLflow
- **Pipeline Orchestration**: Prefect
- **Model Deployment**: Flask + Docker
- **Monitoring**: Evidently
- **Testing**: Pytest
- **CI/CD & Best Practices**: GitHub Actions, pre-commit hooks, Makefile

## 📊 Dataset: RetinaMNIST

- **Task**: Ordinal regression (5-level grading of diabetic retinopathy severity).
- **Image size**: 3 × 224 × 224
- **Samples**: Train (1080), Validation (120), Test (400).
- **License**: CC BY 4.0.
  
## Steps
1. **Training**: `train_pipeline.py` tracks experiments via MLflow.
2. **Deployment**: Containerized Flask API serving predictions.
3. **Monitoring**: `monitoring_flow.py` checks data drift and logs metrics.

## How to Run
```bash
docker build -t retina-app .
docker run -p 8000:8000 retina-app
