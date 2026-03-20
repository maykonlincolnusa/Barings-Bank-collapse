FROM python:3.12-slim

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY app ./app
COPY sources.csv ./
COPY model_card.md ./
COPY data_card.md ./

RUN pip install --no-cache-dir .

EXPOSE 8000 8501

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

