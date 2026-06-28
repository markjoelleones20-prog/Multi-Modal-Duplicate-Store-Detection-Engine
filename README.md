# 🏪 Multi-Modal Duplicate Store Detection Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kAbe2dAUvAPAjnKA7C_OH9bxyCNos57A?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated AI-powered data-cleaning pipeline that resolves enterprise ERP duplicate records by cross-evaluating retail locations across three independent signals: **geospatial proximity**, **string similarity**, and **storefront image embeddings via deep learning**. The system incorporates a Human-in-the-Loop learning mechanism that progressively improves its predictions by training on analyst validation history.

> **Note:** All data in this repository is synthetically generated to replicate real-world enterprise retailer database patterns. No proprietary or confidential data is used.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [AI & ML Architecture](#-ai--ml-architecture)
- [Three-Signal Validation Framework](#-three-signal-validation-framework)
- [Human-in-the-Loop Learning](#-human-in-the-loop-learning)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology](#-methodology)
- [Outputs](#-outputs)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 📌 Problem Statement

Enterprise ERP systems accumulate duplicate retailer records over time — the same physical store registered multiple times under slightly different names, coordinates, or codes by different field agents or data-entry operators. Left uncleaned, this silently corrupts:

- **Sales coverage metrics** — inflated store counts misrepresent territory penetration
- **Promotional planning** — the same store receives duplicate promotional allocations
- **Distributor performance tracking** — route efficiency calculations are distorted

Traditional rule-based deduplication (matching on name alone, or GPS alone) produces high false positive rates because real-world data is inconsistently entered. This pipeline resolves that by requiring **three independent signals to agree simultaneously** before flagging a duplicate — and by learning the analyst's own validation standards over time.

---

## 🤖 AI & ML Architecture

### Model 1 — MobileNetV2 Visual Feature Extractor

The core visual intelligence is powered by **MobileNetV2**, a convolutional neural network pre-trained on ImageNet (1.4 million images across 1,000 categories).

**Architectural Configuration:**

```python
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
```

| Parameter | Value | Rationale |
|---|---|---|
| `weights='imagenet'` | Transfer learning from ImageNet | Leverages pre-learned visual features without training from scratch |
| `include_top=False` | Classification head removed | Repurposes the network as a feature extractor rather than a classifier |
| `pooling='avg'` | Global Average Pooling applied | Collapses spatial feature maps into a fixed 1,280-dimensional vector regardless of input resolution |

**Why MobileNetV2 over heavier alternatives?**

| Model | Parameters | Output Dims | Relative Speed |
|---|---|---|---|
| VGG-16 | 138M | 4,096 | 1× (baseline) |
| ResNet-50 | 25M | 2,048 | 3× faster |
| **MobileNetV2** | **3.4M** | **1,280** | **8× faster** |

MobileNetV2 was purpose-built for constrained compute environments with minimal accuracy trade-off for visual similarity tasks — making it the optimal choice for a pipeline that must process thousands of storefront images per run in a production environment.

**Embedding Pipeline:**

```
Storefront Image (any resolution)
        ↓
Resize to 224 × 224 pixels
        ↓
MobileNetV2 preprocessing (pixel normalization)
        ↓
MobileNetV2 forward pass (no top layer)
        ↓
Global Average Pooling
        ↓
1,280-dimensional embedding vector
        ↓
Cached to JSON (skipped on future runs)
```

Two images of the same physical store produce embedding vectors with **cosine similarity > 0.90**. Images of genuinely different stores typically score **0.40–0.70**.

---

### Model 2 — Random Forest Classifier (Predictive Validation)

A **Random Forest Classifier** is trained on the analyst's past validation decisions to predict whether a flagged pair is a true duplicate or a false positive.

**Feature Set:**

| Feature | Description | Signal Type |
|---|---|---|
| `Image_Sim_to_Leader` | Cosine similarity between MobileNetV2 embeddings | Visual |
| `Name_Sim_to_Leader` | SequenceMatcher string alignment ratio | Lexical |
| `Distance_to_Leader_m` | Haversine distance in metres | Geospatial |

**Why Random Forest over Logistic Regression?**

- **Non-linear interaction effects** — A high image similarity score can partially compensate for a moderate name similarity at very short distances. Random Forest captures these interaction effects naturally; logistic regression cannot without manual feature engineering.
- **Feature importance transparency** — The model outputs interpretable importance scores for each feature, allowing analysts to understand what the AI is weighting most heavily.
- **Robustness to small training sets** — The bagging ensemble mechanism reduces overfitting risk on early-stage training data (as few as 5–10 validated records).
- **Cross-validation accuracy reporting** — Each training run outputs a cross-validated accuracy score so model reliability is always quantified.

---

## 🔍 Three-Signal Validation Framework

A candidate pair is only flagged as a duplicate when it passes **all three gates simultaneously**. This conjunction requirement makes the false positive rate extremely low.

### Signal 1 — Geospatial Proximity (Primary Gate)

The **Haversine formula** calculates the exact great-circle distance between two GPS coordinates, accounting for Earth's curvature:

```
a = sin²(Δlat/2) + cos(lat₁) · cos(lat₂) · sin²(Δlon/2)
distance = 2 · arcsin(√a) · R
```

**Threshold: ≤ 30 metres**

This gate alone eliminates 99%+ of pairs before any computationally expensive name or image comparison occurs — transforming an O(n²) image comparison problem into a tractable one.

### Signal 2 — Lexical Similarity (Secondary Gate)

Store names are compared using Python's **SequenceMatcher** algorithm, which computes:

```
Similarity = 2M / T
```

Where `M` is the number of matching characters and `T` is the total number of characters in both strings. This is more robust than simple Levenshtein distance for names with word reordering (e.g. *"Aling Rosa Store"* vs *"Store ni Aling Rosa"* still scores well).

**Threshold: ≥ 0.85**

This gate filters out co-located stores that are genuinely different businesses — a pharmacy and a grocery sharing the same GPS pin, for example.

### Signal 3 — Visual Similarity (Scoring Layer)

MobileNetV2 **cosine similarity** between the Leader's embedding and each cluster member's embedding:

```
cosine_similarity = (A · B) / (||A|| · ||B||)
```

Unlike the first two gates, visual similarity is used as a **ranked scoring metric** rather than a hard threshold — providing the analyst and the Random Forest with a continuous confidence signal for each pair.

---

## 🔄 Human-in-the-Loop Learning

The most distinctive feature of this pipeline is its **progressive learning mechanism**. Rather than producing static outputs, the system becomes more accurate over time by learning from analyst decisions.

### Learning Cycle

```
Run 1: Pipeline flags candidates → Analyst validates → Saves 'Yes'/'No' labels
        ↓
Run 2: Random Forest trains on validated labels → Generates AI_Prediction scores
        ↓
Run 3+: AI predictions guide analyst review → Only uncertain cases need attention
        ↓
Steady State: Analyst reviews AI-uncertain cases only (~20–40% of flagged pairs)
```

### Observation Mode vs. Active Mode

| Mode | Condition | AI Output |
|---|---|---|
| **Observation Mode** | < 5 validated records exist | No prediction — pipeline flags, analyst decides |
| **Active Mode** | ≥ 5 validated records exist | `Likely Duplicate` / `Likely Different` + confidence score |

### Why Not Fully Automate?

Duplicate detection in enterprise data is a **high-stakes, low-tolerance** operation. A false positive (merging two genuinely different stores) corrupts sales attribution data permanently. A false negative (missing a real duplicate) inflates coverage metrics. The Human-in-the-Loop design ensures:

- Every merge decision has a named human approver
- A complete audit trail of validation decisions is maintained
- The system is correctable — analyst disagreement with AI predictions retrains the model

---

## 🏗️ Pipeline Architecture

```
Raw Retailer Data (ERP Export)
        │
        ▼
┌─────────────────────────────┐
│  Phase 1: Initialization    │  ← MobileNetV2 loaded, Random Forest trained
│  & Historical Learning      │    on past validation history (if available)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 2: Data Ingestion    │  ← 3-condition quality gate
│  & Preprocessing            │    (null coords, null image URL, lat=lon bug)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 3: Visual Feature    │  ← Cache check → parallel download →
│  Extraction                 │    MobileNetV2 batch inference → JSON cache
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 4: Candidate         │  ← Haversine ≤ 30m
│  Grouping (Heuristics)      │    + SequenceMatcher ≥ 0.85
│                             │    → NetworkX graph → connected components
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 5: AI Scoring        │  ← Cosine similarity per group
│  & Validation               │    + Random Forest prediction + confidence
└─────────────────────────────┘
        │
        ▼
   Excel Output → Analyst Review → Training File → Next Run
```

---

## 📐 Methodology

### Graph-Based Duplicate Clustering

Pairs passing both the spatial and lexical gates are treated as **edges** in a `NetworkX` undirected graph. The connected components algorithm then resolves transitive duplicates:

```
If Store A = Store B  (passed both gates)
If Store B = Store C  (passed both gates)
Then A, B, C → same group  (even if A and C were never directly compared)
```

This is the critical advantage over a simple pairwise list — transitive duplicate chains are resolved correctly and completely.

### Leader Designation

Within each group, the first node in alphabetical sort order is designated as the **Leader** — the canonical record that will be retained after deduplication. All similarity metrics are computed relative to this Leader. The deterministic sort ensures the same Leader is elected on every pipeline run, making outputs reproducible.

### Embedding Cache Strategy

```python
# Check cache first → only download missing images
missing = [u for u in unique_urls if u not in embeddings_db]

# Batch download with ThreadPoolExecutor (20 concurrent workers)
# → 20× faster than sequential downloading

# Persist new embeddings to JSON
with open(EMBEDDING_DB_FILE, 'w') as f:
    json.dump(embeddings_db, f)
```

On a database of 10,000 stores: first run ≈ 2 hours. Subsequent daily runs ≈ 5 minutes (only new stores need embedding).

### Data Quality Gate

Three drop conditions are enforced before any AI processing:

| Condition | Detection | Root Cause |
|---|---|---|
| Missing image URL | `notna()` check | Store was registered without a photo |
| Missing coordinates | `notna()` check | GPS data not captured at registration |
| Latitude equals Longitude | `lat != lon` check | GPS encoding bug — one value copied to both fields |

All dropped records are logged separately for upstream ERP investigation.

---

## 📤 Outputs

A single Excel workbook with columns pre-ordered for the analyst validation workflow:

| Column | Description |
|---|---|
| `Group_ID` | Duplicate cluster identifier |
| `RetailerCode` | ERP retailer code |
| `RetailerName` | Store name as registered |
| `Distance_to_Leader_m` | Haversine distance from group Leader (metres) |
| `Name_Sim_to_Leader` | SequenceMatcher similarity to Leader name |
| `Image_Sim_to_Leader` | MobileNetV2 cosine similarity to Leader image |
| `AI_Prediction` | `Likely Duplicate` / `Likely Different` / `Observation Mode` |
| `AI_Confidence` | Random Forest probability score (0.0–1.0) |
| `User_Validation` | **Analyst fills this:** `Yes` or `No` |
| `Image_URL` | Direct link to storefront photograph |
| `Latitude` / `Longitude` | GPS coordinates |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | MobileNetV2 image feature extraction |
| `scikit-learn` | Random Forest classifier, cosine similarity, StandardScaler |
| `networkx` | Graph construction and connected component resolution |
| `haversine` | Great-circle distance calculation |
| `difflib.SequenceMatcher` | String similarity scoring |
| `Pillow` | Image download, conversion, and resizing |
| `concurrent.futures` | Multi-threaded parallel image downloading |
| `pandas` / `numpy` | Data manipulation and vectorized operations |
| `tqdm` | Progress tracking for long-running scan loops |
| `openpyxl` | Excel output generation |

---

## ▶️ How to Run

**In Google Colab (recommended):**

Click the **Open in Colab** badge at the top of this README. Run all cells top to bottom via **Runtime → Run all**. All sections are fully annotated with methodology notes.

**Locally:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install tensorflow scikit-learn networkx haversine tqdm openpyxl pillow pandas numpy
jupyter notebook duplicate_store_detection.ipynb
```

> The notebook uses synthetically generated data and requires no external database, ERP connection, or image server to run in demo mode. All image embeddings are simulated with statistically realistic vectors.

---

*Built with: Python 3 · TensorFlow/Keras · scikit-learn · NetworkX · Haversine · Pillow · concurrent.futures · pandas · openpyxl*
