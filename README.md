# Music-Genre-Classification

Music genre classification project using **Mel Spectrogram images** and deep learning. We trained two hybrid CNN→RNN models with a **multi-head temporal attention** layer, and deployed them in a **Streamlit** web app for interactive predictions. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Overview

Raw audio is high-dimensional and highly temporal, and genre boundaries can be subjective and blurry. This project tackles those challenges by converting audio into **Mel Spectrogram images** (treating audio classification as an image + sequence problem), then learning:

- **Spatial features** from spectrograms with CNNs  
- **Temporal context** with BiLSTMs  
- **Important time steps** with a **multi-head temporal attention** mechanism :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Dataset

We use the **GTZAN** dataset with 10 genres:

`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock` :contentReference[oaicite:4]{index=4}

### Spectrogram format
- Audio → **Mel Spectrogram images**
- Image size: **128 × 256 × 3** :contentReference[oaicite:5]{index=5}

### Split strategy (duplicate-safe)
Train/Val/Test = **70% / 15% / 15%** with **file hashing** to keep duplicates (or near-duplicates) within the same split and avoid leakage. :contentReference[oaicite:6]{index=6}

---

## System Pipeline

1. **Data Preparation**: GTZAN audio → Mel spectrogram images  
2. **Preprocessing & Augmentation**: resize, normalize, augment  
3. **Model Training**: train **Model A** and **Model B**  
4. **Evaluation**: standard metrics + **Test-Time Augmentation (TTA)**  
5. **Application**: Streamlit app to upload images and get predictions :contentReference[oaicite:7]{index=7}

---

## Models

### Model A — Custom CNN + BiLSTM + Attention
**CNN backbone** (input: 128×256×3) with 4 conv blocks:
- Conv2D → BatchNorm → Dropout → MaxPool  
- Squeeze-and-Excitation (SE) blocks in later stages  
- Global Average Pooling → Dense head with L2 regularization :contentReference[oaicite:8]{index=8}

Then:
- CNN output reshaped into a sequence
- **Two stacked BiLSTM layers** (+ BatchNorm)
- **Multi-head temporal attention (4 heads)**
- Softmax classifier  
Saved as a full model: `.h5` :contentReference[oaicite:9]{index=9}

### Model B — EfficientNet-B0 (Transfer Learning) + BiLSTM + Attention
- Input: 128×256×3 Mel spectrogram images
- **EfficientNet-B0 (ImageNet)** as a feature extractor (fine-tuned)
- Regularization: Gaussian noise + dropout + L2 dense layers
- Two BiLSTMs (160, 80 units)
- **Multi-head temporal attention (4 heads)**
- Softmax classifier  
Saved as weights-only: `.weights.h5` :contentReference[oaicite:10]{index=10}

---

## Temporal Attention (Multi-Head)

Custom Keras layer:
- Takes LSTM outputs over time
- 4 attention heads learn different temporal patterns
- Each head: Dense(tanh) + projection → attention weights
- Produces a context vector for classification :contentReference[oaicite:11]{index=11}

---

## Augmentation & Regularization

### Augmentation
- Brightness & contrast jitter
- Horizontal flip (time reversal simulation)
- SpecAugment-style masking (time/frequency masking) :contentReference[oaicite:12]{index=12}

### Normalization
- Model A: `[0, 1]`
- Model B: `[0, 255]` :contentReference[oaicite:13]{index=13}

### Training tools
- Adam optimizer
- ReduceLROnPlateau (factor 0.5 after 8 epochs w/o val acc improvement)
- EarlyStopping (after 30 epochs w/o improvement)
- SparseCategoricalCrossentropy + label smoothing (0.1) :contentReference[oaicite:14]{index=14}

---

## Results

From the reported test results:

- **Model A**: 72.00% accuracy (73.33% with TTA)  
- **Model B**: 74.00% accuracy (75.33% with TTA)  

Model B consistently outperformed Model A, and TTA improved both models. :contentReference[oaicite:15]{index=15}

---

## Streamlit App

A Streamlit demo app (`appnoda.py`) provides:
- Upload spectrogram images (PNG/JPG/JPEG)
- Choose Model A / Model B / Both
- Probability distribution over 10 genres
- Side-by-side comparison and agreement status :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

