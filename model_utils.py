import io
import sys

import torch
import mlflow
import torch.nn as nn
import torch.optim as optim
import mlflow.pytorch
from prefect import flow, task
from medmnist import RetinaMNIST
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Fix UnicodeEncodeError for emoji in MLflow output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retina-mnist-experiment")


@task(name='load_data', log_prints=True)
def load_data(batch_size=32):
    # pylint:disable=import-outside-toplevel
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = RetinaMNIST(split="train", download=True, transform=transform)
    val_dataset = RetinaMNIST(split="val", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


@task(name='initialize_model', log_prints=True)
def initialize_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)  # RetinaMNIST has 5 labels
    return model


@task(name='train_model', log_prints=True)
def train_model(model, train_loader, epochs=2, device="cpu", lr=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("model_architecture", "resnet50")
        mlflow.log_param("dataset", "RetinaMNIST")

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.squeeze().long().to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")
            mlflow.log_metric("loss", running_loss / len(train_loader), step=epoch)
        mlflow.pytorch.log_model(model, "model")
    return model


@task(name='evaluate_model', log_prints=True)
def evaluate(model, val_loader, device="cpu"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(val_loader), 100 * correct / total


@task(name='save_model', log_prints=True)
def save_model(model, path=r"model\resnet50_retinamnist.pth"):
    torch.save(model.state_dict(), path)


@flow(name='RetinaMNIST-Training-Pipeline', retry_delay_seconds=10, retries=3)
def train_retina_mnist():
    train_loader, val_loader = load_data()
    model = initialize_model()
    trained_model = train_model(model, train_loader)
    val_loss, val_accuracy = evaluate(trained_model, val_loader)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    save_model(trained_model)


if __name__ == "__main__":
    train_retina_mnist.serve(name="my-mlopsproject-deployment", cron="0 0 * * *")

    train_retina_mnist()
