import torch
import mlflow
import torch.nn as nn
import mlflow.pytorch
import torchvision.models as models

# Load model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(
    torch.load("resnet50_retinamnist.pth", map_location=torch.device('cpu'))
)
model.eval()

# Dummy metric (replace with real eval logic later)
val_accuracy = 0.52
loss = 2.09

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("project-experiment")


with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "resnet50")
    mlflow.log_param("input_size", 224)
    mlflow.log_param("dataset", "RetinaMNIST")

    # Log metrics
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("loss", loss)

    # Log model
    mlflow.pytorch.log_model(model, name="model")

    # Optionally log artifacts like plots or configs
    # mlflow.log_artifact("config.yaml")
