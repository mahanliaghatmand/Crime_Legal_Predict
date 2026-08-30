# ⚖️ Crime & Legal Text Classification — Model Monitoring Dashboard

A 4-class crime/legal text classification project using four different Keras architectures (Dense, Conv1D, LSTM, BiLSTM), paired with a professional **Streamlit** monitoring dashboard for tracking model performance, data drift, error analysis, and prediction explainability.

## 🎯 Output Classes

| Class | Label |
|---|---|
| 0 | No crime |
| 1 | Violent crime / drugs |
| 2 | Theft / cybercrime |
| 3 | Corruption / financial crime |

## 🧠 About the Model

### Dataset
- **520 train / 111 validation / 112 test** samples (742 total), split with stratified sampling (`random_state=42`) so class proportions are preserved across splits.
- Strong class imbalance: **377** "No Crime" · **208** "Violent Crime / Drugs" · **130** "Theft / Cybercrime" · **28** "Corruption / Financial Crime" samples.
- Class imbalance is handled at training time with `sklearn`'s `compute_class_weight("balanced", ...)`, so minority classes contribute proportionally more to the loss.

### Text Preprocessing
- Text is tokenized with Keras' `TextVectorization` layer: **max vocabulary = 5,000 tokens**, **fixed sequence length = 32**, lowercased and punctuation-stripped, split on whitespace.
- On the training data this produced a vocabulary of **418 tokens** (including the reserved empty and `[UNK]` tokens) — a very small vocabulary, since the dataset itself is small.
- The fitted vocabulary is saved to `artifacts/vectorizer_vocab_crime_model.pkl` and reloaded at inference time so the app doesn't need to re-fit it.

### Architectures
All four models share the same skeleton — a 64-dim `Embedding` layer feeding into an architecture-specific encoder, then `Dense(64, relu) → Dropout(0.3) → Dense(4, softmax)` — so the comparison isolates the effect of the encoder:

| Model | Encoder | Idea |
|---|---|---|
| **Dense** | `GlobalAveragePooling1D` over embeddings | Bag-of-words style baseline; ignores word order |
| **Conv1D** | `Conv1D(64, kernel=3)` + `GlobalMaxPooling1D` | Picks up local 3-word patterns/phrases |
| **LSTM** | `LSTM(64)` | Sequential model reading the text left to right |
| **BiLSTM** | `Bidirectional(LSTM(64))` | Reads the text in both directions for fuller context |

**Training setup**: Adam optimizer (`lr=1e-3`), sparse categorical cross-entropy loss, batch size 16, up to 30 epochs with `EarlyStopping` (patience 5, restores best weights) and `ReduceLROnPlateau` (patience 3, factor 0.5) both monitoring validation loss.

### Results on the Test Set (112 samples)

| Model | Accuracy | F1 (weighted) | Precision (weighted) | Recall (weighted) |
|---|---|---|---|---|
| Dense | **1.000** | 1.000 | 1.000 | 1.000 |
| Conv1D | 0.866 | 0.871 | 0.902 | 0.866 |
| BiLSTM | 0.661 | 0.581 | 0.727 | 0.661 |
| LSTM | 0.527 | 0.381 | 0.549 | 0.527 |

> ⚠️ **Take these numbers with a grain of salt.** The dataset has only 742 rows total and one class (`Corruption / Financial Crime`) has just 28 examples — 4 of them in the test set. Dense's perfect 1.000 score, on a tiny vocabulary (418 tokens) with such a small test set, is a strong sign of overfitting/memorization rather than genuine generalization, not evidence that a bag-of-words model beats recurrent ones. The recurrent models (LSTM/BiLSTM) likely underperformed because there isn't enough data to learn useful sequential patterns — with this little data they'd benefit from more regularization, pretrained embeddings, or simply more training examples. The dashboard's **Data Drift** and **Error Analysis** tabs are there precisely to help catch this kind of fragility once the model sees new, more diverse text.

### Inference
At prediction time, the app vectorizes input text with the saved vocabulary and runs it through the selected model to get a softmax probability over the 4 classes; the highest-probability class is shown along with its confidence. The **Explainability** tab additionally runs an occlusion test — it re-predicts the text with one word zeroed out at a time and measures how much the target class's confidence drops, to highlight which words drove the decision.

## ✨ Dashboard Features

- **Overview**: Accuracy / Precision / Recall / F1 on the test set + simulated production throughput
- **Confusion Matrix** (raw and normalized) with a detailed per-class report
- **Data Drift Detection**: compares text-length and class-distribution shifts between reference (train) and production data using a KS test, plus a drift-over-time trend chart
- **Error Analysis**: filter low-confidence samples and misclassified predictions
- **Alert History**: automatic daily alerts for accuracy drops, low confidence, data drift, and traffic spikes
- **Filter, Search & Export**: filter by class / confidence range / text search, plus CSV and summary report downloads
- **Explainability**: occlusion-based analysis showing each word's contribution to the model's decision

## 🗂️ Project Structure

```
.
├── app.py                                   # Streamlit app
├── requirements.txt
├── dataset/
│   └── dataset.csv                          # training data (text + label)
├── artifacts/
│   ├── vectorizer_vocab_crime_model.pkl     # TextVectorization vocabulary
│   ├── model_Dense_crime.keras
│   ├── model_Conv1D_crime.keras
│   ├── model_LSTM_crime.keras
│   └── model_BiLSTM_crime.keras
└── crime_legal_text_classification_keras.ipynb   # training notebook
```

> ⚠️ File paths in `app.py` must exactly match this folder layout (`dataset/` and `artifacts/`) — otherwise the app crashes on startup with a `FileNotFoundError`.

## 🚀 Run Locally

```bash
git clone https://github.com/<username>/Crime_Legal_Predict.git
cd Crime_Legal_Predict

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (the `.keras` model files can be a few MB each; if the total size gets large, use **Git LFS**).
2. On [share.streamlit.io](https://share.streamlit.io), click **New app** and select the repo, branch, and `app.py` as the entry point.
3. Before deploying, open **Advanced settings** on that same screen and set **Python version** to **3.11** (see below).
4. Click Deploy.

## 🐍 Setting the Python Version on Streamlit

A `runtime.txt` file is no longer a reliable way to pin the Python version on Streamlit Community Cloud — per recent user reports and issues in Streamlit's own repo, it's frequently ignored, and the Cloud defaults to the latest Python version, which can be incompatible with TensorFlow.

The correct, current approach:
- When creating the app, open the **Advanced settings** dialog and pick the Python version from the dropdown.
- For this project (`tensorflow>=2.16`), **Python 3.11** is recommended (3.12/3.13 often work with newer TensorFlow too, but 3.11 is currently the most stable choice).
- Note: the Python version **cannot be changed after deployment** — to change it you must delete the app and redeploy with the new version selected.

## 🐛 Bug Fixed in `app.py`

**Wrong data/model paths** — the code was looking for `data/` and `models/` folders, while in this repo the data lives at `dataset/dataset.csv` and the vectorizer/model files live under `artifacts/`. This mismatch caused the app to fail immediately with a "failed to load models or data" error. Paths were corrected to:

```python
DATA_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
VOCAB_PATH = os.path.join(BASE_DIR, "artifacts", "vectorizer_vocab_crime_model.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts")
```

**Notes to avoid future errors:**
- The installed TensorFlow version must support the `.keras` (Keras 3) format; `tensorflow>=2.16` in `requirements.txt` covers this.
- If the combined size of the `.keras` files exceeds GitHub's 100 MB per-file limit, use Git LFS.
- `vectorizer_vocab_crime_model.pkl` must match the exact settings used in `app.py` (`max_tokens=5000`, `output_sequence_length=32`). If the model is retrained with different settings, update these values in `app.py` too.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `Streamlit` · `Plotly` · `scikit-learn` · `pandas` / `numpy` · `scipy`

## 📄 About

This is a machine learning project, built for educational/portfolio purposes.
