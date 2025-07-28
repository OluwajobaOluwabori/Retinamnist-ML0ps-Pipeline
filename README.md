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
- **MLOps best practices** (tests, pre-commit hooks, Makefile, linter).

## 📂 Project Structure

```text
├── models/ # Saved models (ResNet-50 .pth)
├── app.py # Flask API for serving predictions
├── monitoring_flow.py # Prefect flow for model/data monitoring
├── model_utils.py # Utilities for model training pipeline and tracking with MLflow
├── Dockerfile # Container setup for Flask API
├── Makefile # Build, test, and deployment commands
├── docker-compose.yaml     # Compose setup for Grafana, Postgres, Adminer
├── tests/ # Unit & integration tests
│ ├── test_model.py → Unit Test
│ └── test_app.py → Integration Test
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## 🛠 Tech Stack

- **ML Framework**: PyTorch
- **Experiment Tracking and Model Registry**: MLflow
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

## ⚙️ Setup Instructions
1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/retina-mlops-project.git
   cd retina-mlops-project
   ```

2. **Create & activate a Conda env**
     ```bash
     conda create -n retina-mlops python=3.10 -y
     conda activate retina-mlops
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install pre‑commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🏋️‍♀️ Model Training & Tracking
Run and deploy the Prefect training flow (which logs to MLflow and Prefect) with schedule to run every hour:
- MLflow UI:
   ```bash
   mlflow ui --port 5000
   ```
- Prefect
   ```bash
  prefect server start
   ```

  Once it’s running:
   ```bash
   python model_utils.py
   ```

Open http://127.0.0.1:5000 to explore the experiments on MLFlow UI.

## 🌐 Deployment
Containerized Flask API serving predictions.
1. Run Flask locally
   ```bash
   python app.py
   ```
2. Dockerize & run
   ```bash
   docker build -t retina-app .
   docker run -p 8000:8000 retina-app
   ```
API endpoint: http://127.0.0.1:5001/predict
CURL: curl -X POST http://localhost:8000/predict -F "file=@img.jpg"

## 🔍 Monitoring
To run the Prefect monitoring flow (Evidently + PostgreSQL + alerting)-checks data drift and logs metrics:
   ```bash
   python monitoring_flow.py
   ```
- Reports saved under reports/
- Metrics stored in PostgreSQL → visualise in Grafana

 - Access the database: http://localhost:8080/?pgsql=db
 - Access grafana: http://localhost:3000/


## ✅ Testing
- Run tests:
   ```bash
   pytest tests/ -v
   ```

- Lint & format:
   ```bash
   black .
   isort .
   pylint --recursive=y .
   ```
- Makefile shortcuts:
   ```bash
   make test
   make lint
   make build
   ```

## How to Run
```bash
docker build -t retina-app .
docker run -p 8000:8000 retina-app
curl -X POST http://localhost:8000/predict -F "file=@img.jpg"
```
