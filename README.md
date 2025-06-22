# Basic-Design-2025
## Instructor: Prof. Abolghasem Sadeghi-Niaraki
### Key Task: Evaluating Design Alternatives
### Group 8
### Team Members: 
Nazmul Hassan Rafiyun(24012965)/ **Team Leader**,
MIFTE MEHEDI HASAN(25013445),
ARIFUL ISLAM(24013592),
FATTAKHOV OYBEK RUSTAM UGLI(22013110),
BAKYTOV YELNUR (23013078).

# Jet Engine Health Monitoring System 🛩️

A predictive maintenance system for jet engines using the NASA C-MAPSS dataset. This project builds and compares machine learning and deep learning models to estimate the Remaining Useful Life (RUL) of jet engines, and visualizes engine health through an interactive dashboard.

---

## 🚀 Features

- Data loading and exploration
- Feature engineering including moving averages and rates of change
- RUL label generation
- Normalization and sequence preparation
- Model 1: Random Forest Regressor
- Model 2: LSTM Neural Network
- Model comparison with evaluation metrics
- Interactive health monitoring dashboard for a random engine

---


---

## 🧠 Dataset

- **Source:** [NASA C-MAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **Subset Used:** FD003 (with 3 operational settings, 21 sensors)

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/rafiiowa/Basic-Design-2025.git
cd Jet Engine Health Monitoring
pip install -r requirements.txt



## 📁 Project Structure

jet-engine-health-monitoring/
│
├── data/                          # Raw and processed data files (DO NOT push large files to GitHub)
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   ├── RUL_FD003.txt
│
├── notebooks/
│   └── Main_Project.ipynb         # Your original Colab notebook
│
├── src/                           # Modular Python scripts 
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── dashboard.py
│   └── utils.py
│
├── outputs/                       # Model outputs, plots, logs (optional)
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and instructions
└── .gitignore                     # Files/folders to ignore in Git

