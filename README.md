<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/Hraj24/NETRA-2.0-trustworthy-secure-nwdaf-mlops/main/reports/shap_summary_fl.png" alt="NETRA 2.0 Banner" width="100%" style="border-radius: 8px;" />
</p> -->

# 🛡️ NETRA 2.0: Trustworthy & Secure NWDAF MLOps Platform

NETRA (Network Traffic Risk Analyzer) is an MLOps platform built around the 3GPP **NWDAF** (Network Data Analytics Function) framework, upgraded for **6G IMT-2030** compliance. It delivers network traffic load predictions with built-in safeguards for production machine learning systems.

[![Python Version](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.123-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Flower](https://img.shields.io/badge/Flower-1.24-FF6F61?logo=flower&logoColor=white)](https://flower.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📖 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Docker Compose Deployment](#docker-compose-deployment)
  - [Local Development Setup](#local-development-setup)
- [6G IMT-2030 Compliance](#-6g-imt-2030-compliance)
  - [Usage Scenarios](#usage-scenarios)
  - [Feature Schema](#12-imt-2030-features)
- [API Documentation](#-api-endpoints)
- [Workflow Execution](#-data-generation--pipelines)
  - [Data Pipeline](#data-generation)
  - [Federated Learning Training](#federated-learning-training)
- [System Safeguards](#-key-components--safeguards)
  - [Drift Detection Matrix](#hybrid-drift-detection)
  - [Model Rollback Mechanism](#model-rollback--auto-recovery)
- [Tech Stack](#-tech-stack)
- [Standards & Compliance](#-standards-compliance)

---

## 🌟 Key Features

* **IMT-2030 Alignment**: Full support for 6G usage scenarios and 12 baseline KPIs following ITU-R M.2160 specifications.
* **Privacy-Preserving Federated Learning**: Localized model training using Flower (FedAvg strategy) prevents raw telemetry exposure.
* **Concurrent Concept Drift Detection**: Moving Average, ADWIN, and DDM run in parallel to monitor data variance.
* **Self-Healing Model Lifecycle**: Automated rollback to stable models on SLA drift and self-recovery after stabilization.
* **Explainable Network Decisions**: Local and global feature attribution calculations using SHAP linear explainers.
* **Real-Time Observability UI**: Interactive telemetry inputs, SHAP visualization panels, and drift indicators.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    NETRA 2.0 Platform                    │
├────────────────────────┬─────────────────────────────────┤
│   React Dashboard      │     FastAPI Backend             │
│   (Port 3000)          │     (Port 8000)                 │
│                        │                                 │
│  ┌──────────────┐      │  ┌───────────┐ ┌────────────┐  │
│  │ IMT-2030     │──────┼─▶│ /predict  │ │ /explain   │  │
│  │ Input Form   │      │  │ /health   │ │            │  │
│  │ 6G Scenarios │◀─────┼──│           │ │            │  │
│  │ 12 KPIs      │      │  └─────┬─────┘ └─────┬──────┘  │
│  │ SHAP Panel   │      │        │              │         │
│  │ Status Panel │      │  ┌─────▼──────────────▼──────┐  │
│  └──────────────┘      │  │    Model Manager           │  │
│                        │  │  (rollback / auto-recover) │  │
│                        │  └─────┬─────────────────────┘  │
│                        │        │                        │
│                        │  ┌─────▼─────────────────────┐  │
│                        │  │  Hybrid Drift Detectors    │  │
│                        │  │  MA · ADWIN · DDM          │  │
│                        │  └───────────────────────────┘  │
│                        │                                 │
│                        │  ┌───────────────────────────┐  │
│                        │  │  FL Global Model (SGD)     │  │
│                        │  │  12 IMT-2030 features      │  │
│                        │  │  trained via Flower FedAvg │  │
│                        │  └───────────────────────────┘  │
└────────────────────────┴─────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
NETRA-2.0/
├── src/                          # Python backend source (IMT-2030)
│   ├── api.py                    # FastAPI application layer
│   ├── data_pipeline.py          # Data synthesis and validation pipeline
│   ├── drift.py                  # Moving Average drift detector
│   ├── drift_adaptive.py         # ADWIN and DDM drift detectors
│   ├── drift_logger.py           # Audit logger for concept drift events
│   ├── rollback.py               # ModelManager for fallback and recovery
│   ├── explain.py                # SHAP explanation and attribution engine
│   ├── shap_logger.py            # Local record of SHAP features
│   ├── fl_server.py              # Federated Learning server orchestrator
│   ├── fl_client.py              # Domain-specific training node
│   ├── train_centralized.py      # Baseline centralized training
│   ├── train_and_save_models.py  # Model serialization workflow
│   └── models.py                 # Neural Network and Random Forest models
│
├── nwdaf-dashboard/              # React frontend (Vite environment)
│   ├── src/
│   │   ├── App.jsx               # Tab-based dashboard entry point
│   │   ├── api.js                # Network client interface
│   │   └── components/
│   │       ├── InputForm.jsx     # Scenario data entry (12 KPIs)
│   │       ├── PredictionPanel.jsx
│   │       ├── ExplainPanel.jsx  # SHAP visualizer components
│   │       └── StatusPanel.jsx   # Platform health and drift display
│   └── Dockerfile                # Production Nginx multi-stage configuration
│
├── models/                       # Serialized model artifacts
│   ├── fl_global_model.pkl       # FL aggregated network model
│   ├── model_v1.pkl              # Trusted baseline deployment
│   └── model_v2_bad.pkl          # Deliberately degraded test model
│
├── data/                         # 6G Network datasets
│   ├── traffic_synthetic.csv     # Global aggregated collection
│   ├── domain_a.csv              # IC & HRLLC network data
│   ├── domain_b.csv              # MC & UC network data
│   └── domain_c.csv              # AIAC & ISAC network data
│
├── reports/                      # Static attribution plots
├── logs/                         # File logging directories
├── Dockerfile.backend            # Uvicorn FastAPI build
├── docker-compose.yml            # Multi-container orchestrator
├── requirements-backend.txt      # Backend library requirements
├── requirements-dev.txt          # Development tools
└── .github/workflows/
    └── docker-ci.yml             # Automatic build validation
```

---

## 🚀 Getting Started

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and active.
* [Git](https://git-scm.com/) source control tool.
* [Python 3.10+](https://www.python.org/) for execution without containers.
* [Node.js 18+](https://nodejs.org/) for local UI compilation.

### Docker Compose Deployment
Execute the following commands to initialize all services automatically in container environments:

```bash
# Clone the codebase
git clone https://github.com/Hraj24/NETRA-2.0-trustworthy-secure-nwdaf-mlops.git
cd NETRA-2.0-trustworthy-secure-nwdaf-mlops

# Build and execute containers in background mode
docker compose up --build -d
```

| Service | Access Endpoint | Documentation |
|:---|:---|:---|
| **FastAPI Backend** | [http://localhost:8000](http://localhost:8000) | Interactive Swagger interface at [/docs](http://localhost:8000/docs) |
| **Dashboard UI** | [http://localhost:3000](http://localhost:3000) | React-based monitoring portal |

### Local Development Setup
To execute development servers on your local operating system:

#### 1. Backend API Service
```bash
# Set up Python virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On Linux/MacOS:
source .venv/bin/activate

# Install package dependencies
pip install -r requirements-backend.txt

# Start backend execution
uvicorn src.api:app --reload --port 8000
```

#### 2. Frontend Dashboard React App
```bash
# Move to the dashboard root
cd nwdaf-dashboard

# Install node dependencies
npm install

# Execute development environment
npm run dev
```

---

## 🧠 6G IMT-2030 Compliance

Aligned with the **ITU-R Recommendation M.2160** framework, the system is designed around modern 6G usage scenarios.

### Usage Scenarios

| Scenario | Definition | Network Focus Area | Predecessor |
|:---:|:---|:---|:---|
| **IC** | Immersive Communication | High bandwidth requirements | 5G eMBB |
| **HRLLC** | Hyper Reliable & Low Latency | Mission-critical response profiles | 5G URLLC |
| **MC** | Massive Communication | High device density networks | 5G mMTC |
| **UC** | Ubiquitous Connectivity | Global universal access targets | *New Scenario* |
| **AIAC** | AI and Communication | Intelligent compute distribution | *New Scenario* |
| **ISAC** | Integrated Sensing & Communication | Hardware sensing collaboration | *New Scenario* |

### 12 IMT-2030 Features
Predictions are generated using 12 network features:

1. `time_of_day`: Hour indicator (integer from `0` to `23`).
2. `usage_scenario`: 6G profile identifier (one of: `IC`, `HRLLC`, `MC`, `UC`, `AIAC`, `ISAC`).
3. `throughput_mbps`: Realized packet delivery speed (Mbps).
4. `latency_ms`: One-way plane packet latency (ms).
5. `jitter_ms`: Microsecond-level packet jitter variation (ms).
6. `packet_loss_rate`: Decimal probability of packet drop.
7. `reliability_target`: Target delivery SLA (%).
8. `connection_density_km2`: Device distribution density (units/km²).
9. `mobility_kmph`: Client transit speed (km/h).
10. `area_traffic_capacity_score`: Capacity rating factor (`0.0` - `1.0`).
11. `ai_load_score`: Computational compute profile load (`0.0` - `1.0`).
12. `resilience_score`: Architectural fault-tolerant system metric (`0.0` - `1.0`).

---

## 🔌 API Endpoints

### 1. Predict Traffic Load
* **Route**: `POST /predict`
* **Content-Type**: `application/json`

**Sample Payload**:
```json
{
  "time_of_day": 12,
  "usage_scenario": "IC",
  "throughput_mbps": 400.0,
  "latency_ms": 3.0,
  "jitter_ms": 1.5,
  "packet_loss_rate": 0.0001,
  "reliability_target": 99.999,
  "connection_density_km2": 500000,
  "mobility_kmph": 100,
  "area_traffic_capacity_score": 0.35,
  "ai_load_score": 0.35,
  "resilience_score": 0.75
}
```

**Sample Response**:
```json
{
  "predicted_future_load": 275.4,
  "explanation": "Network conditions appear stable with moderate predicted load.",
  "model_version": "FL-v2.0-IMT2030 (normal)",
  "warning": null
}
```

### 2. Attributable Feature Explanations
* **Route**: `POST /explain`
* **Content-Type**: `application/json`

**Sample Response**:
```json
{
  "prediction": 275.4,
  "shap_values": {
    "time_of_day": 0.12,
    "usage_scenario": -0.05,
    "throughput_mbps": 1.85,
    "latency_ms": -0.23,
    "jitter_ms": 0.04,
    "packet_loss_rate": 0.0,
    "reliability_target": 0.01,
    "connection_density_km2": 0.44,
    "mobility_kmph": -0.12,
    "area_traffic_capacity_score": 0.88,
    "ai_load_score": 1.05,
    "resilience_score": -0.32
  },
  "note": "SHAP values indicate per-feature contribution to the prediction"
}
```

### 3. Service Diagnostics
* **Route**: `GET /health`
* **Function**: Obtains deployment metrics, active classifier version, drift details, and current rollback targets.

---

## 🧪 Data Generation & Pipelines

### Data Generation
Execute these scripts to synthesize and validate IMT-2030 telemetry arrays:

```bash
# Generate full telemetry set containing 5000 records
python -m src.data_pipeline --generate --samples 5000 --seed 42

# Run compliance check on existing dataset file
python -m src.data_pipeline --validate --input data/traffic_synthetic.csv

# Output isolated multi-domain partition subsets
python -m src.data_pipeline --domains --samples 2500 --seed 42
```

Outputs are exported directly into the `data/` folder:
* `data/traffic_synthetic.csv` (global dataset)
* `data/domain_a.csv` (immersive and hyper-reliable scenarios)
* `data/domain_b.csv` (massive and ubiquitous scenarios)
* `data/domain_c.csv` (artificial intelligence and sensing scenarios)

### Federated Learning Training
Preserve localized domain data by training the global target model using Flower nodes:

```bash
# 1. Start the central aggregator process (Terminal 1)
python -m src.fl_server

# 2. Run local clients for respective network domains (Terminals 2, 3, 4)
python -m src.fl_client --domain domain_a
python -m src.fl_client --domain domain_b
python -m src.fl_client --domain domain_c
```
The aggregated output parameters are serialized as `models/fl_global_model.pkl`.

---

## 🔑 Key Components & Safeguards

### Hybrid Drift Detection
The platform evaluates input profiles concurrently across three drift monitors:

| Detection System | Algorithmic Approach | SLA Responsibility |
|:---|:---|:---|
| **Moving Average** | Sliding Window comparison metrics | Initiates recovery system actions |
| **ADWIN** | Adaptive Window size estimation | Early alarm trigger indicator |
| **DDM** | Error Rate statistical validation | Trend degradation alert tracker |

### Model Rollback & Auto-Recovery

```
   [ Telemetry Flow ]
           │
     ┌─────▼─────┐
     │ Prediction│
     └─────┬─────┘
           │
           ├───────────────────────────────┐
           ▼                               ▼
    [ Normal Performance ]           [ SLA Drift Detected (MA) ]
           │                               │
    Keep Primary Model              Rollback to Stable Model
           │                               │
    Audit System Ok                 Maintain 5 stable windows
           │                               │
     ┌─────▼─────┐                         ▼
     │ End Step  │                  Auto-Recovery to Primary
     └───────────┘
```

* **Drift Mitigation**: When sliding window performance drifts past defined SLAs, FastAPI redirects requests to `model_v1.pkl`.
* **System Healing**: System monitors telemetry. Following 5 stable sliding window intervals, the framework recovers to the main deployment target.
* **Audit Trail**: Operational transitions are recorded under `drift_events.csv`.

---

## 🛠️ Tech Stack

* **Service Core**: Python 3.10, FastAPI, Uvicorn, Pydantic.
* **Statistical Modeling**: scikit-learn, SHAP, NumPy, Pandas.
* **Telemetry Drift**: River (ADWIN, DDM), custom metrics algorithms.
* **Federated Engines**: Flower (flwr framework).
* **Operational Dashboard**: React 18, Vite environment, Recharts graph elements.
* **Infrastructure Packaging**: Docker and Compose profiles.

---

## 📜 Standards Compliance

NETRA 2.0 is aligned with the following standardization structures:
* **ITU-R Recommendation M.2160** (September 2023): Framework and overall objectives of the development of IMT for 2030 and beyond.
* **ITU-R IMT-2030 Technical Performance Requirements** (February 2026): Performance specifications for the 6G radio interface framework.

---
<p align="center">
  NETRA 2.0: Trustworthy MLOps for Future Network Analytics.
</p>
