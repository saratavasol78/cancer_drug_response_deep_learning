# CIFAR-10 Image Classification with CNN (TensorFlow/Keras)

![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)

A production-style computer vision project: a custom VGG-style CNN and a MobileNetV2
transfer-learning model, with evaluation reports, Grad-CAM explainability,
a REST API (FastAPI) and Docker support.

## Features
- **Two models**: custom CNN (from scratch) and MobileNetV2 (transfer learning + fine-tuning)
- **Config-driven**: every hyperparameter lives in `configs/config.yaml`
- **Full evaluation**: accuracy, per-class report, normalized confusion matrix
- **Explainability**: Grad-CAM heatmaps to see what the CNN looks at
- **Serving**: FastAPI endpoint + Dockerfile
- **Engineering**: unit tests (pytest), linting (ruff), CI (GitHub Actions), Makefile

## Project structure
```
├── configs/config.yaml        # hyperparameters and paths
├── src/cnn_project/
│   ├── model.py               # custom CNN + MobileNetV2 builders
│   ├── data.py                # tf.data pipeline (train/val/test)
│   ├── train.py               # training (+ fine-tuning), callbacks
│   ├── evaluate.py            # metrics, confusion matrix
│   ├── predict.py             # CLI inference (+ Grad-CAM)
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── samples.py             # export sample images from the test set
│   └── api.py                 # FastAPI app
├── tests/                     # pytest unit tests
├── Dockerfile
├── Makefile
└── .github/workflows/ci.yml
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # or: pip install -r requirements-dev.txt && pip install -e .
```

## Train
```bash
make train            # custom CNN
make train-transfer   # MobileNetV2 (frozen backbone, then fine-tuning)
```
Artifacts are saved in `artifacts/<model_name>/`: `best.keras`, `training_curves.png`,
`confusion_matrix.png`, `classification_report.txt`, `metrics.json`, `history.json`
and TensorBoard logs (`tensorboard --logdir artifacts`).

Override options from the CLI, e.g. `python -m cnn_project.train --model custom_cnn --epochs 5`.

## Results
Fill this table with your own numbers after training (see `metrics.json`).

| Model | Test accuracy |
|-------|---------------|
| Custom CNN | _run `make train`_ |
| MobileNetV2 (fine-tuned) | _run `make train-transfer`_ |

## Predict and explain
```bash
make samples                                   # creates samples/*.png
python -m cnn_project.predict samples/cat.png --gradcam
```
The Grad-CAM overlay is saved in `outputs/` (works with the custom CNN).

## REST API
```bash
make serve
curl -X POST -F "file=@samples/cat.png" "http://127.0.0.1:8000/predict?top_k=3"
```
Interactive docs: http://127.0.0.1:8000/docs

## Docker
```bash
make docker-build
make docker-run       # mounts ./artifacts into the container
```

## Tests and lint
```bash
make test
make lint
```

## Ideas for the future
- Hyperparameter search with Keras Tuner
- Export to TFLite / ONNX
- Track experiments with MLflow or Weights & Biases
- Train on a custom dataset (e.g. medical images) by replacing `data.py`

## License
MIT
