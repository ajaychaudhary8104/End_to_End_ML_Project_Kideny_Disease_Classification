# Kidney Disease Classification - MLflow & DVC

## 📋 Project Overview

This is an end-to-end machine learning project for kidney disease classification using Convolutional Neural Networks (CNN). The project leverages **MLflow** for experiment tracking and **DVC** (Data Version Control) for reproducible ML pipelines. The model classifies CT scan images as either **Normal** or **Tumor** cases.

### 🎯 Key Features

- **Deep Learning Pipeline**: VGG16-based CNN architecture for medical image classification
- **Experiment Tracking**: MLflow integration for tracking metrics, parameters, and models
- **Data Versioning**: DVC for tracking dataset versions and pipeline reproducibility
- **Web Application**: FastAPI and Flask-based REST APIs for model serving
- **Modular Architecture**: Clean separation of concerns with components, pipelines, and utilities
- **Configuration Management**: YAML-based configuration for easy customization

---

## 📁 Project Structure

```
├── artifacts/                          # Model artifacts and preprocessed data
│   ├── data_ingestion/                # Raw and processed datasets
│   │   └── kidney-ct-scan-image/
│   │       ├── Normal/               # Normal kidney images
│   │       └── Tumor/                # Tumor kidney images
│   ├── prepare_base_model/           # Base model files
│   │   ├── base_model.keras
│   │   └── base_model_updated.keras
│   └── training/                      # Trained model
│       └── model.keras
│
├── src/cnnClassifier/                 # Main package
│   ├── components/                    # Core ML components
│   │   ├── data_ingestion.py         # Data loading and preprocessing
│   │   ├── prepare_base_model.py     # Model initialization
│   │   ├── model_training.py         # Training logic
│   │   └── model_evaluation_mlflow.py # Evaluation with MLflow
│   │
│   ├── config/
│   │   └── configuration.py           # Configuration management
│   │
│   ├── entity/
│   │   └── config_entity.py          # Entity definitions
│   │
│   ├── pipeline/                      # ML pipeline stages
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py              # Inference pipeline
│   │
│   └── utils/
│       └── common.py                  # Utility functions
│
├── config/
│   └── config.yaml                    # Configuration file
│
├── research/                          # Jupyter notebooks for exploration
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
│
├── templates/
│   └── index.html                     # Web UI template
│
├── app.py                             # Flask application
├── app_fastapi.py                     # FastAPI application
├── main.py                            # Main entry point
├── dvc.yaml                           # DVC pipeline configuration
├── params.yaml                        # Model hyperparameters
├── config.yaml                        # Project configuration
└── requirements.txt                   # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Git

### Installation Steps

#### STEP 1: Clone the Repository

```bash
git clone https://github.com/ajaychaudhary8104/End_to_End_ML_Project_Kideny_Disease_Classification.git
cd End_to_End_ML_Project_Kideny_Disease_Classification
```

#### STEP 2: Create Conda Environment

```bash
conda create -n cnncls python=3.12 -y
conda activate cnncls
```

#### STEP 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### STEP 4: Run the Application

```bash
# Flask application
python app.py

# Or FastAPI application
python app_fastapi.py
```

#### STEP 5: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📊 Pipeline Workflow

The project follows a structured ML workflow:

1. **Data Ingestion** - Load and extract kidney CT scan images
2. **Base Model Preparation** - Initialize VGG16 pre-trained model
3. **Model Training** - Fine-tune the model on kidney disease data
4. **Model Evaluation** - Evaluate performance with MLflow tracking
5. **Prediction** - Deploy model for inference on new images

### Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages

---

## 🔧 Configuration Files

### config.yaml
Contains paths to artifacts, source, and output directories:
```yaml
artifacts_root: artifacts
source_download_dirs:
  - data
data_ingestion_config:
  unzip_dir: artifacts/data_ingestion
```

### params.yaml
Defines model hyperparameters:
```yaml
epochs: 25
batch_size: 16
learning_rate: 0.001
```

---

## 🤖 ML Experiment Tracking with MLflow

### MLflow Features

- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage models
- **Production Deployment**: Serve models in production

### Starting MLflow UI

```bash
mlflow ui
```

Access MLflow UI at: `http://localhost:5000`

### DagsHub Integration

Track experiments on DagsHub:

```bash
set MLFLOW_TRACKING_URI=<your_dagshub_url>
set MLFLOW_TRACKING_USERNAME=<your_username>
set MLFLOW_TRACKING_PASSWORD=<your_password>
python main.py
```

---

## 📦 DVC (Data Version Control)

### Initialize DVC

```bash
dvc init
```

### Run DVC Pipeline

```bash
dvc repro
```

### View Pipeline DAG

```bash
dvc dag
```

### Common DVC Commands

```bash
dvc add <file_or_directory>        # Track data with DVC
dvc push                            # Push data to remote storage
dvc pull                            # Pull data from remote storage
dvc status                          # Check pipeline status
```

---

## 📡 API Endpoints

### Flask Application (app.py)

- **Home Page**: `GET /`
- **Upload & Predict**: `POST /predict` (with image file)
- **Results**: `GET /results`

### FastAPI Application (app_fastapi.py)

- **API Docs**: `GET /docs`
- **Predict**: `POST /predict` (with image file)
- **Health Check**: `GET /health`

---

## 🧪 Jupyter Notebooks

Research notebooks for exploration and experimentation:

- `research/01_data_ingestion.ipynb` - Explore data loading
- `research/02_prepare_base_model.ipynb` - Test model preparation
- `research/03_model_training.ipynb` - Training experiments
- `research/04_model_evaluation_with_mlflow.ipynb` - Evaluation with MLflow

---

## 📋 Requirements

See `requirements.txt` for all dependencies:

```
tensorflow>=2.13.0
scikit-learn
mlflow
dvc
fastapi
uvicorn
flask
pillow
numpy
pandas
pyyaml
```

---

## 📚 Documentation

### MLflow
- [Official Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tutorial](https://youtu.be/qdcHHrsXA48)

### DVC
- [Official Website](https://dvc.org/)
- [DVC Documentation](https://dvc.org/doc)

### DagsHub
- [DagsHub Platform](https://dagshub.com/)

---

## 🏗️ Project Architecture

The project uses a modular architecture:

- **Components**: Individual steps of the ML pipeline
- **Pipeline**: Orchestrate components in sequence
- **Config**: Centralized configuration management
- **Entity**: Data classes for type safety
- **Utils**: Shared utility functions

---

## 📝 Model Details

**Architecture**: VGG16 (Pre-trained on ImageNet)
- Transfer Learning approach for medical image classification
- Fine-tuned on kidney CT scan images
- Input Shape: (224, 224, 3)
- Output: Binary classification (Normal vs Tumor)

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

---

## 📄 License

This project is licensed under the LICENSE file in the repository.

---

## 👨‍💻 Author

For questions or support, please open an issue in the repository.










# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app