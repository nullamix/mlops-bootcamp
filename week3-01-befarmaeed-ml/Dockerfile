FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update &&     apt-get install -y --no-install-recommends build-essential &&     rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip &&     pip install --prefix=/install --no-cache-dir -r requirements.txt &&     pip install --prefix=/install --no-cache-dir .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "airbnb_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
