#  MLOps Project 

##  Overview

Ce projet implémente un pipeline MLOps complet couvrant :

-  Versioning des données avec **DVC**
-  Entraînement & optimisation des modèles (LogReg, RandomForest)
-  Orchestration avec **ZenML**
-  Tracking des expériences avec **MLflow**
-  Serving via **FastAPI** (v1 / v2)
-  Monitoring avec **Prometheus** & **Grafana**
-  Interface utilisateur avec **Streamlit**
-  Déploiement sur **Azure Container Apps**
-  Infrastructure avec **Docker** & **Docker Compose**


---

##  Structure du projet

```
mlops_Project/
│
├── api/
│   ├── main.py
│   ├── metrics.py
│   ├── v1/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   └── v2/
│       ├── router.py
│       ├── schemas.py
│       └── service.py
│
├── training/
│   └── src/
│       ├── zenml_step/
│       └── zenml_pipelines/
│
├── streamlit.py
│   
│   
│
├── data/
│   └── raw/
│
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.api
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

## 🐳 Lancement de l'infrastructure

### 1️ Build & démarrage des services

```bash
docker compose up -d --build
```

**Services lancés :**
- MinIO
- MLflow
- ZenML Server
- Training container
- FastAPI
- Prometheus
- Grafana

---

##  Gestion des données (DVC)

```bash
rm data/raw/breast_cancer.csv
dvc pull
```

---

## 🧪 Entraînement des modèles (container training)

```bash
docker exec -it training bash
```

### Entraînement classique MLflow

```bash
python training/src/train.py --model rf
python training/src/train.py --model logreg
```

### Optimisation avec Optuna

```bash
python training/src/optuna_study.py
```

---

##  Pipelines ZenML

### Connexion au serveur ZenML

```bash
zenml connect --url http://zenml-server:8080
```

### Lancer un pipeline

```bash
python training/src/run_pipeline_baseline.py
```

### Vérifier les runs

```bash
zenml pipeline runs list
```

---

## 🪣 Buckets MinIO (création MANUELLE)

| Bucket | Utilisation |
|--------|-------------|
| `zenml-artifacts` | ZenML |
| `mlflow-artifacts` | MLflow |
| `dvcstore` | DVC |

---

##  Configuration ZenML / MLflow

### Enregistrement du tracker MLflow

```bash
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri=http://mlflow:5000 \
  --tracking_token="dummy-token"
```

### Secret S3

```bash
zenml secret create aws_s3_secret \
  --aws_access_key_id="XXX" \
  --aws_secret_access_key="XXX" \
  --aws_session_token="XXX"
```

### Artifact Store

```bash
zenml artifact-store register s3_artifacts \
  --flavor=s3 \
  --path='s3://project-mlops1/zenml-artifacts' \
  --authentication_secret=aws_s3_secret
```

### Stack ZenML

```bash
zenml orchestrator register local_orch --flavor=local

zenml stack register mlflow_stack \
  -o local_orch \
  -a s3_artifacts \
  -e mlflow_tracker

zenml stack set mlflow_stack
```

---

##  API d'inférence (FastAPI)

### Lancer l'API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Swagger local

[http://localhost:8000/docs](http://localhost:8000/docs)

### ☁️ API déployée sur Azure

🔗 **Swagger Azure**

[https://fastapi-app.yellowwater-2f47f3a8.francecentral.azurecontainerapps.io/docs](https://fastapi-app.yellowwater-2f47f3a8.francecentral.azurecontainerapps.io/docs)

### Exemple de requête v1

```bash
curl -X POST http://localhost:8000/api/v1/predict \
-H "Content-Type: application/json" \
-d '{"features":[0,0,0,...]}'
```

### Exemple v2

```bash
curl -X POST http://localhost:8000/api/v2/predict
```

---

##  Monitoring – Prometheus & Grafana

### Endpoint métriques

```bash
curl http://localhost:8000/metrics
```

### Prometheus UI

[http://localhost:9090](http://localhost:9090)

**Exemples de requêtes :**
- `REQUEST_COUNT`
- `REQUEST_LATENCY`

### Grafana UI

[http://localhost:3000](http://localhost:3000)

**Login par défaut :**
- Username: `admin`
- Password: `admin`

---

##  Interface Streamlit (Inference UI)

### Lancer Streamlit

```bash
streamlit run streamlit.py
```

**Fonctionnalités :**
- Choix model v1 ou v2
- Appel API Azure
- Visualisation des prédictions
- Démo A/B testing

---

##  Objectifs MLOps atteints

- ✔️ Versioning données & modèles
- ✔️ Pipelines reproductibles
- ✔️ Tracking & artifacts
- ✔️ Serving multi-versions
- ✔️ Monitoring par version
- ✔️ UI utilisateur
- ✔️ Déploiement Cloud

---

##  Auteur

**Zaineb Ben Fadhl**  
Étudiante en Génie Informatique – Data Science & AI

🔗 **GitHub** : [https://github.com/zaineb-ben-fadhl](https://github.com/zaineb-ben-fadhl)

🔗 **API Azure** : [https://fastapi-app.yellowwater-2f47f3a8.francecentral.azurecontainerapps.io/docs](https://fastapi-app.yellowwater-2f47f3a8.francecentral.azurecontainerapps.io/docs)