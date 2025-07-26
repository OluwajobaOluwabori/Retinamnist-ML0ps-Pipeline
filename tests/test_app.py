import pytest

from app import app


@pytest.fixture
def client():
    '''This creates a test client that mimics requests to your Flask app, without running the real server.'''
    app.testing = True  # Flask runs in test mode (no debugging)
    return app.test_client()


def test_health_check(client):
    '''This tests the root endpoint to ensure the app is running.'''
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome" in response.data


def test_predict(client):
    with open("tests/img.jpg", "rb") as img_file:
        response = client.post('/predict', data={'file': img_file})

    assert response.status_code == 200
    data = response.get_json()
    assert 'predicted_label' in data
    assert isinstance(data['predicted_label'], int)
    assert 0 <= data['predicted_label'] < 5  # Assuming 5 classes in RetinaMNIST


def test_predict_no_file(client):
    response = client.post('/predict')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No file part in the request'
