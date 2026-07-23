# VadAnuvaadNLP: Punjabi-English Neural Machine Translation (NMT)

A complete, end-to-end Machine Translation project fine-tuning **Meta's NLLB-200 (600M Distilled)** model on the **Anuvaad Parallel Corpus** (1.7+ million sentence pairs), optimized for **100,000 sampled sentence pairs** trained on **Kaggle GPU**.

---

## 📌 Features
- **Cleaned Data Subset Engine**: Preprocesses 1.7M raw sentence pairs, applies Unicode NFC normalization (crucial for Gurmukhi script consistency), deduplicates, and extracts 100k high-quality sentence pairs.
- **Kaggle GPU Fine-Tuning Pipeline**: Self-contained Jupyter Notebook (`notebooks/punjabi_english_nmt_kaggle.ipynb`) designed for free NVIDIA T4/P100 GPUs.
- **Automated Metric Evaluation**: Evaluates translation quality using **sacreBLEU** and **chrF++**.
- **Interactive Gradio Web App**: Local web interface (`app.py`) for live Punjabi ↔ English translations.

---

## 📁 Repository Directory Structure

```
VadAnuvaadNLP/
│── Anuvaad.en-pa.en         # Raw English Corpus (1.7M lines)
│── Anuvaad.en-pa.pa         # Raw Punjabi Corpus (1.7M lines)
│── punjabi_english_100k.csv # Preprocessed 100k dataset (generated)
│── README.md                # Project documentation & step-by-step guide
│── requirements.txt         # Python dependencies
│── config.yaml              # Hyperparameters & path configurations
│── app.py                   # Gradio Web Interface for Live Translation
│── src/
│   ├── preprocess.py        # Data cleaning, deduplication & sampling script
│   └── utils.py             # Utility functions
└── notebooks/
    └── punjabi_english_nmt_kaggle.ipynb # Kaggle GPU training notebook
```

---

## 🚀 Quick Start Guide

### Step 1: Preprocess Data Locally
Run the standalone preprocessing script to extract and clean 100,000 sentence pairs from the raw Anuvaad corpus:

```bash
python src/preprocess.py
```

This will create `punjabi_english_100k.csv` (~18–20 MB) containing 85% Train, 7.5% Validation, and 7.5% Test splits.

---

### Step 2: Fine-Tune on Kaggle GPU (15–30 Minutes)

1. Open **[Kaggle](https://www.kaggle.com)** and create a **New Notebook**.
2. Upload `punjabi_english_100k.csv` as a Dataset on Kaggle (or upload directly into the notebook sidebar).
3. Import or copy the contents of `notebooks/punjabi_english_nmt_kaggle.ipynb` into Kaggle.
4. Enable **GPU Acceleration** in Kaggle settings:
   - Click `Settings` ➔ `Accelerator` ➔ Select **GPU T4 x2** or **GPU P100**.
5. Click **Run All** (`Shift + Enter`).
6. After training completes (~2-3 epochs), download `punjabi_english_nllb_final.zip` from Kaggle's `/kaggle/working/` output directory.

---

### Step 3: Run the Web UI Application

1. Install local dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Unzip your trained model checkpoint (or use default NLLB model weights).
3. Launch the Gradio Web App:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:7860` in your web browser to test Punjabi ↔ English translations!

---

## 📊 Evaluation & Metrics
| Metric | Description | Target Score |
| :--- | :--- | :--- |
| **sacreBLEU** | Standard NMT n-gram precision metric | > 25.0 |
| **chrF / chrF++** | Character-level n-gram F-score (ideal for Indic languages) | > 50.0 |

---

## 📜 Dataset Citation & License
This project utilizes the **Anuvaad Parallel Corpus** from **OPUS**:
- License: CC-BY-4.0
- Citation: J. Tiedemann, 2012, *Parallel Data, Tools and Interfaces in OPUS*. In Proceedings of LREC 2012.
