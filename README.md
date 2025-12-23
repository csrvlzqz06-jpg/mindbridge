# MindBridge

MindBridge is an AI-based early risk detection engine designed to support student well-being through ethical, non-invasive academic time-series analysis. The system analyzes longitudinal learning behavior signals to identify early patterns associated with academic disengagement and potential mental health risk, without accessing private content or personal communications.

This project is developed using Huawei AI technologies, with MindSpore as the core deep learning framework, and is designed to scale on Huawei Cloud and ModelArts.

---

## Problem Statement

Educational institutions worldwide face increasing challenges related to student mental health, academic burnout, and disengagement. Current detection methods are often reactive, subjective, or rely on invasive surveys and manual reporting, which limits early intervention.

There is a strong need for a scalable, data-driven, and privacy-preserving system that enables educators to identify early warning signals and take preventive actions before critical academic or emotional deterioration occurs.

---

## Solution Overview

MindBridge addresses this challenge by introducing a time-series AI engine that analyzes weekly academic behavior indicators and classifies students into three risk levels:

- LOW risk  
- MEDIUM risk  
- HIGH risk  

The system focuses on trends and temporal patterns rather than isolated events, enabling early detection and continuous monitoring.

---

## Key Features

- Non-invasive analysis (no private messages, content, or personal data)
- Time-series modeling using LSTM neural networks
- Explainable risk classification with probability distribution
- Interactive dashboard for educators
- Scalable cloud-ready architecture
- Ethical-by-design approach for educational environments

---

## Data Design

The MVP uses structured numerical time-series signals that simulate realistic academic platform behavior, including:

- Assignment completion rate  
- Average daily login time  
- Days active per week  
- Login time variance  
- Schedule irregularity  
- Self-reported stress indicator  

Each student sample represents an 8-week temporal window.

Synthetic data is used in this stage to validate the complete AI pipeline while ensuring privacy and compliance.

---

## Model Architecture

The core model is an LSTM-based neural network implemented in MindSpore:

- Input: (8 weeks × 6 features)
- Temporal feature extraction via stacked LSTM layers
- Fully connected classification head
- Softmax output for probabilistic risk estimation

The model is trained and evaluated using MindSpore training and checkpoint mechanisms.

---

## Inference and Visualization

A Streamlit-based MVP dashboard demonstrates:

- Risk classification per student sample
- Confidence scores and probability distribution
- Weekly signal visualization for interpretability
- Comparison between LOW, MEDIUM, and HIGH risk profiles

This interface represents the educator-facing layer of MindBridge.

---

## Ethical Considerations

MindBridge does not diagnose medical conditions and is not a replacement for professional mental health evaluation.

The system is designed to:
- Respect student privacy
- Avoid invasive data collection
- Support educators as a decision-support tool
- Enable preventive and supportive interventions

---

## Technology Stack

- MindSpore (deep learning framework)
- Python
- NumPy
- Streamlit (MVP dashboard)
- Huawei Cloud (planned deployment)
- ModelArts (training and scaling – regional stage)

---

## Project Status

- MVP completed and functional
- End-to-end AI pipeline validated
- Ready for national competition submission
- Cloud deployment planned for regional stage

---

## Disclaimer

This project uses synthetic and public-style numerical signals for demonstration purposes. It does not process real student data in its current form.
