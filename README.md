# 🏋️ AI Fitness & Diet Recommendation System

A modern, full-stack fitness and nutritional strategy application utilizing a **decoupled microservices architecture**. The system captures user health metrics via an interactive client interface, routes data payloads across a local network, and processes predictions using machine learning models served through an independent backend engine.

---

## 🏗️ System Architecture Overview

The system is engineered as two completely separated, modular services to achieve platform independence and production scalability:
1. **Frontend Client Interface (`app.py`):** Built using **Streamlit**, functioning as a pure presentation layer. It captures comprehensive user parameters, validates data inputs, handles background API routing over HTTP POST, and renders custom `matplotlib` charts and diagnostic tables upon receiving response vectors.
2. **Computational Machine Learning Engine (`main.py`):** Built using **FastAPI** and served via **Uvicorn**. It hosts data transformations and houses model evaluation routines, isolating core calculations away from the client interface.

┌─────────────────────────┐               ┌─────────────────────────┐
│   Frontend Client UI    │  HTTP POST    │   Backend ML Engine     │
│    (Streamlit App)      ├──────────────►│    (FastAPI Server)     │
│                         │  JSON Data    │                         │
│  - User Input Forms     │               │  - Linear Regression    │
│  - Matplotlib Graphics  │◄──────────────┤  - Random Forest        │
│  - Diagnostic Tables    │  JSON Results │  - Cosine Similarity    │
└─────────────────────────┘               └─────────────────────────┘


---

## 🤖 Machine Learning Pipeline Specifications

The decoupled backend trains and optimizes four distinct predictive systems to generate tailored individual roadmaps:
* **Target Calorie Predictor:** Implements **Linear Regression** evaluating continuous user variables (Age, BMI, Height, and Active Parameters) to predict precise dietary baseline thresholds.
* **Workout Optimization Classifier:** Uses a **Random Forest Classifier** trained over collective dataset traits to map workout labels.
* **Nutritional Menu Formulator:** Implements text feature vector tracking via **Cosine Similarity** mapping client goals and medical limitations against a baseline profile database to extract high-matching diet schedules.
* **Progress Trend Tracker:** Executes an interactive sliding-window projection timeline simulation modeling time-series weight trends across weekly intervals.

---

## 📂 Repository Structure

```text
├── app.py              # High-fidelity Streamlit frontend interface & chart engine
├── requirements.txt    # Production package dependencies manifest
└── README.md           # Technical project documentation overview

⚡ Quick Start & Deployment Guide
🚀 Local Production Deployment (Network Host Mode)
To run the decoupled configuration locally and test across external network client devices (such as a mobile phone):
Find your local network IP address:
Open Terminal and run ipconfig (Windows) or ifconfig (Mac/Linux) to locate your IPv4 address (e.g., 192.168.1.5).
Launch the backend server engine:

Bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Launch the frontend dashboard client:

Bash
python -m streamlit run app.py
Access cross-platform:
Open any browser window on a mobile device connected to the same Wi-Fi router and navigate to http://<YOUR_LAPTOP_IP>:8501.

🛠️ Built With
Frontend View: Streamlit, Matplotlib, Seaborn
API Networking Core: FastAPI, Uvicorn, Requests, Pydantic
Machine Learning Pipeline: Scikit-Learn, Pandas, NumPy, OpenPyXL
Developed as an Academic Milestone Project for the Athenura Professional Internship Program.
