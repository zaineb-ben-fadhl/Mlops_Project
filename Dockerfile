FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système (nécessaires pour sklearn, dvc)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances
COPY requirements.txt .

# Installer les libs Python
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copier le projet
COPY . .

# Commande par défaut
CMD ["bash"]
