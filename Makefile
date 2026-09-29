.PHONY: install train train-transfer evaluate samples test lint serve docker-build docker-run

install:
	pip install -r requirements-dev.txt
	pip install -e .

train:
	python -m cnn_project.train --model custom_cnn

train-transfer:
	python -m cnn_project.train --model mobilenet_v2

evaluate:
	python -m cnn_project.evaluate --model-path artifacts/custom_cnn/best.keras

samples:
	python -m cnn_project.samples

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn cnn_project.api:app --reload

docker-build:
	docker build -t cifar10-cnn .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts cifar10-cnn
