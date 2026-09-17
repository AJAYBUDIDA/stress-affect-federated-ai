# AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI

## Overview
This project is a software-based simulation for real-time stress and affect detection utilizing **Federated Learning (FL)** and **Explainable AI (XAI)** on the **WESAD** (Wearable Stress and Affect Detection) dataset.

### Core Architecture & Features
- **Dataset**: WESAD (Wearable Stress and Affect Detection)
- **Target Classes**: Neutral, Stress, Amusement
- **Federated Learning**: Simulated across 3 subject-based clients using Flower (`flwr`)
- **Machine Learning / Deep Learning**: PyTorch (`torch`)
- **Explainable AI**: Model interpretability using SHAP (`shap`)
- **Web Dashboard**: Interactive real-time simulation dashboard built with Flask and modern frontend visualization technologies.

## Project Structure
```text
stress-affect-federated-ai/
│
├── data/
│   ├── raw/          # Original WESAD dataset files
│   ├── processed/    # Preprocessed feature tables & signals
│   └── clients/      # Data partitioned for 3 FL subject clients
│
├── models/           # Saved PyTorch model checkpoints & global weights
│
├── src/              # Core ML, FL simulation, signal processing & SHAP code
│
├── dashboard/
│   ├── templates/    # Flask HTML templates
│   └── static/       # Static assets (CSS, JS, images)
│       ├── css/
│       ├── js/
│       └── images/
│
├── results/
│   ├── plots/        # Model evaluation & performance metrics plots
│   └── xai/          # SHAP summary & force plots
│
├── requirements.txt  # Project dependencies
├── README.md         # Project documentation
└── run.py            # Main application entry point
```
