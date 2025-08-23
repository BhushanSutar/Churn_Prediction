# 📊 Customer Churn Prediction using ANN

This project is an **Artificial Neural Network (ANN)** based model to predict customer churn. It helps businesses understand whether a customer is likely to leave the service based on various features.  

The model is deployed using **Streamlit**, and preprocessing steps such as **scaling, label encoding, and one-hot encoding** are included for consistency between training and deployment.

---

## 🚀 Features
- Trained on customer churn dataset using **Artificial Neural Network (ANN)**.  
- Includes preprocessing steps:
  - **Standard Scaler** (`scaler.pkl`)
  - **Label Encoder** (`label_encoder.pkl`)
  - **One-Hot Encoder** (`one_hot_encoder.pkl`)
- Deployment with **Streamlit** for interactive prediction.  
- User-friendly UI to input customer details and get real-time churn predictions.  

---

## 📂 Project Structure
```bash
├── churn.ipynb              # Jupyter Notebook with training code
├── app.py                   # Streamlit app for deployment
├── scaler.pkl               # Standard Scaler used during training
├── label_encoder.pkl        # Label Encoder used during training
├── one_hot_encoder.pkl      # One-Hot Encoder used during training
├── Churn_Modelling.h5           # Trained ANN model
└── README.md                # Project Documentation

Model Details

Architecture: Artificial Neural Network (ANN)

Layers: Input → Hidden Layers (ReLU activations) → Output (Sigmoid)

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

Author

BHushan Sutar

GitHub: BhushanSutar

LinkedIn:(https://www.linkedin.com/in/bhushansutar/)
