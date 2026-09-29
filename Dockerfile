FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .

ENV MODEL_PATH=/app/artifacts/custom_cnn/best.keras
EXPOSE 8000

CMD ["uvicorn", "cnn_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
