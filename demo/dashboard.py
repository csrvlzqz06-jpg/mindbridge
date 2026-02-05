import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

import mindspore as ms
from mindspore import Tensor

from model.lstm_model import LSTMRiskNet

CLASSES = ["LOW", "MEDIUM", "HIGH"]
FEATURES = [
    "avg_daily_login_time",
    "login_time_variance",
    "days_active_per_week",
    "assignment_completion_rate",
    "schedule_irregularity",
    "late_submission_rate",
]

st.set_page_config(page_title="MindBridge Demo", layout="centered")
st.title("🧠 MindBridge — MVP Demo")
st.caption("Early risk detection from non-invasive academic time-series signals (MindSpore).")

st.sidebar.header("Paths")
npz_path = st.sidebar.text_input("Dataset (.npz)", "data/windows_dataset.npz")
ckpt_path = st.sidebar.text_input("Checkpoint (.ckpt)", "model/mindbridge_lstm.ckpt")

@st.cache_data
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    return X, y

X, y = load_npz(npz_path)

st.write(f"Loaded dataset: **X={X.shape}**, **y={y.shape}**")
st.write(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Model
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
_, T, F = X.shape
net = LSTMRiskNet(input_size=F, hidden_size=64, num_layers=1, num_classes=3)
ms.load_checkpoint(ckpt_path, net=net)
net.set_train(False)

st.sidebar.header("Select sample")
idx = st.sidebar.number_input("Sample index", min_value=0, max_value=int(X.shape[0]-1), value=0, step=1)

# Inference
x = Tensor(X[idx:idx+1])               # (1, 8, 6)
logits = net(x).asnumpy()[0]           # (3,)
probs = np.exp(logits - logits.max())
probs = probs / probs.sum()
pred = int(probs.argmax())
conf = float(probs[pred])

st.subheader("Prediction")
st.metric("Risk Class", CLASSES[pred], f"confidence {conf:.3f}")
st.write("Probabilities:", {CLASSES[i]: float(probs[i]) for i in range(3)})

st.subheader("Weekly signals (normalized)")
df = pd.DataFrame(X[idx], columns=FEATURES)
df["week"] = np.arange(T)
df = df.set_index("week")
st.line_chart(df)

st.info(
    "This MVP uses synthetic/public-style numeric signals to validate the full MindSpore pipeline. "
    "It does not diagnose medical conditions or analyze private content."
)
