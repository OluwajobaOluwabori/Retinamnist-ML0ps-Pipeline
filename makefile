build:
	docker build -t retina-api .

run:
	docker run -p 8000:8000 retina-api

test:
	curl -X POST http://localhost:8000/predict -F "file=@img.jpg"
	curl -X POST http://localhost:8000/predict -F "file=@img2.jpg"
