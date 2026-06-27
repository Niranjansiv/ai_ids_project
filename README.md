# 🛡️ AI-Driven Network Intrusion & Anomaly Behavioral Detection System

## 📌 Overview
A machine learning-powered security system that monitors network traffic in real time,
detects intrusions, and identifies anomalous behavioral patterns using AI techniques
including supervised learning, reinforcement learning, and explainable AI (XAI).

## 🎯 Motivation
Traditional signature-based Intrusion Detection Systems (IDS) fail to detect 
zero-day attacks and evolving threats. This system bridges that gap by learning 
normal network behavior and flagging deviations — enabling proactive threat 
detection without relying on known attack signatures.

## ✨ Features
- 🔍 Real-time network traffic analysis and classification
- 🤖 ML-based intrusion detection (Random Forest + Deep Learning)
- 📊 Anomaly detection using behavioral baseline modeling
- 🧠 Explainable AI (XAI) to interpret model decisions
- 🎮 Reinforcement Learning agent for adaptive threat response
- 🌐 Detection of unknown/zero-day attack patterns
- 📈 Live visualization dashboard for monitoring

## 🏗️ Architecture
The system is structured into the following modules:

| Module | Description |
|---|---|
| `data/` | Raw and processed network traffic datasets |
| `models/` | Trained ML models for intrusion classification |
| `rl_agent/` | Reinforcement Learning agent for adaptive response |
| `xai/` | Explainability module using SHAP/LIME |
| `presentation/` | Visualizations and demo notebooks |
| `run_pipeline.py` | End-to-end pipeline runner |
| `unknown_attack.py` | Zero-day / unknown attack detection |

## 🧰 Tech Stack
- **Language:** Python
- **ML Libraries:** Scikit-learn, TensorFlow / PyTorch
- **XAI:** SHAP / LIME
- **RL Framework:** OpenAI Gym / Stable-Baselines3
- **Visualization:** Matplotlib, Seaborn
- **Dataset:** CICIDS / NSL-KDD / Custom network traffic data

## ⚙️ Installation
```bash
git clone https://github.com/Niranjansiv/ai_ids_project.git
cd ai_ids_project
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Run the full detection pipeline
python run_pipeline.py

# Detect unknown/zero-day attacks
python unknown_attack.py

# Train the RL agent
python rl_agent/train_agent.py
```

> **Note:** The trained model (`rf_model.pkl`) and processed dataset
> (`processed_data.csv`) are not included due to size constraints.
> Run `run_pipeline.py` to regenerate them locally.

## 📊 Results
- ✅ High accuracy in classifying known attack types
- ✅ Successfully detects anomalous behavior deviating from baseline
- ✅ RL agent adapts responses based on threat severity
- ✅ XAI module provides human-readable explanations for detections

## 🙋‍♂️ Author
**Niranjan Siv**  
[GitHub](https://github.com/Niranjansiv)
