# Semiconductor Predictive Maintenance & AI Failure Prediction

This repository contains an end-to-end Machine Learning pipeline built for semiconductor manufacturing quality assurance. Using Scikit-Learn, the system trains a predictive classification model to detect microchip hardware failures before physical damage occurs based on real-time thermal and electrical telemetry.

## 🚀 Key Features & AI Architecture
* **Hardware Telemetry Simulation:** Generates realistic multi-parameter stress test datasets (Core Temperature, Voltage, and Current) using NumPy.
* **Supervised Machine Learning (Random Forest):** Trains an ensemble classification model (`RandomForestClassifier`) on historical wafer test data to learn hidden physical failure thresholds without hardcoded rules.
* **Model Evaluation & Metrics:** Achieves **89.40% prediction accuracy** on unseen test data, validated through comprehensive Precision, Recall, and F1-Score classification reports.
* **Real-Time Inference Engine:** Evaluates brand-new untested microchips and outputs actionable pass/scrap decisions alongside exact failure probability percentages.

## 🛠️ Tech Stack & Dependencies
* **Core Environment:** Python 3.14+ / Jupyter Notebook (`.ipynb`)
* **Machine Learning:** Scikit-Learn (`train_test_split`, `RandomForestClassifier`, `classification_report`)
* **Data Processing:** Pandas, NumPy
