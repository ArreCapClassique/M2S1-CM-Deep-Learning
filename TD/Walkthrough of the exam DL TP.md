Below is a practical way to decide “which model to use” that matches how the course frames it: **start from the data type + the task output structure**, then choose the **family of architectures that is specialized for that structure**, and only then tune details (depth, units, regularization, etc.). The professor explicitly insists that *architecture choice is hard, often research-level*, but in practice you should map a business/TP context to the right “standard” architecture rather than inventing one from scratch.

---

## 1) The decision procedure (what you do every time)

### Step A — Identify the “data nature”
Ask: *What is one training example?*  
This determines which inductive bias you want the network to have.

1. **Tabular features**  
   One row = one independent example; no spatial or temporal structure.  
   Examples: customer attributes, engineered features, classical ML tables.  
   → Typical bias needed: none special → **fully-connected layers (MLP)**.

2. **Images / grids / spatial tensors**  
   Height × width × channels.  
   Examples: photos, frames, medical imagery.  
   → Need locality + translation robustness → **convolution**.

3. **Sequences / time series / ordered tokens**  
   Order matters, length may vary.  
   Examples: bike rentals by time, text characters/words, protein sequences.  
   → Need order + dependency on the past (and sometimes future) → **recurrent or attention-based models**.

Core course summary: **CNNs for images**, **RNN-family for sequences/time**, **MLP for tabular**.

---

### Step B — Identify the “task type” (what the output looks like)

1. **Regression**  
   Output is continuous (a real number).

2. **Classification**  
   Output is a discrete class (binary or multiclass).

3. **Structured prediction**  
   Output is aligned with the input structure (per pixel, per timestep, per token).

The lectures systematically associate tasks with architectures:
- segmentation → U-Net  
- object detection → YOLO  
- image classification → CNN / ResNet  
- sequence tasks → RNN/LSTM/GRU or Transformers  

---

### Step C — Identify the “input–output topology”
This step is crucial for sequence problems.

- **Many-to-one**  
  Sequence in → one label/value out  
  Example: sentiment of a whole sentence.

- **Many-to-many**  
  Sequence in → sequence out  
  Example: label each timestep/token.

- **One-to-many**  
  One input → sequence output  
  Example: caption generation.

This distinction is explicitly taught with RNN architectures and applications.

---

### Step D — Choose the “standard architecture family”

Once A + B + C are clear, the family is usually determined:

- **Tabular data** → **MLP**  
- **Image classification** → **CNN / ResNet**  
- **Time series / sequences** → **LSTM / GRU** (or Transformer)  
- **Image segmentation** → **U-Net**  
- **Object detection** → **YOLO**  
- **Image generation** → **GANs**  
- **Language / modern sequence modeling** → **Transformers / GPT-type models**

Key teaching point: the *pipeline* (preprocessing, evaluation, data framing) matters as much as the model itself.

---

### Step E — Deciding the *details* (after the model family is fixed)

Once the **architecture family** is fixed (MLP vs CNN vs LSTM/GRU/Transformer), the remaining choices are **not arbitrary**. They fall into **five systematic categories**, and in the TPs you are expected to justify *why* each choice is coherent with the problem.

> Key idea from the course:  
> **The model is only one element of the pipeline; performance depends on representation, training strategy, and evaluation discipline.**

---

### E-1) Input representation & preprocessing (what you feed the model)

#### 1. Encoding of raw inputs
This depends entirely on the chosen family.

###### a) Sequences (LSTM / GRU / Transformer)
Examples: protein sequences (TP 2022–2023), time series (TP 2024–2025)

- **Categorical sequences** (amino acids, characters):
  - integer encoding + embedding layer  
  - or one-hot encoding (simple but higher dimensional)

  Justification (course logic):
  - embeddings allow the model to learn similarity relations between symbols
  - one-hot is acceptable for small vocabularies and pedagogical clarity

- **Numerical sequences** (bike rentals, sensors):
  - normalization or standardization (mean–variance scaling)
  - possibly separate scaling per feature

  Justification:
  - recurrent models are sensitive to scale
  - stabilizes gradient flow and convergence

###### b) Images (CNN)
- pixel normalization (e.g., divide by 255)
- optional data augmentation (rotation, crop, flip)

Justification:
- CNN assumes local correlations; augmentation improves invariance

---

#### 2. Sequence length & windowing
This is **mandatory** in sequence TPs.

- Fixed-length windows (e.g., length = 50 or 100)
- Padding + masking if variable length is allowed

Justification:
- GPUs require fixed tensor shapes
- window length reflects how much past context you assume is useful

**Typical TP sentence**:
> “We restrict sequences to length = 50 in order to control computational cost while preserving sufficient contextual information for prediction.”

---

### E-2) Architecture depth, width, and internal structure

Once you chose *LSTM* (for example), you still must decide **how many layers and units**.

#### 1. Number of layers (depth)
- 1 layer → captures simple dependencies
- multiple stacked layers → hierarchical temporal representations

In TP 2024–2025, this is **explicitly imposed**:
> 4 LSTM layers with dropout between layers

Justification:
- deeper recurrent stacks can model more complex temporal dynamics
- dropout is required to avoid overfitting due to depth

---

#### 2. Number of units per layer (width)
- More units → higher capacity, higher risk of overfitting
- Fewer units → lower capacity, underfitting risk

Typical reasoning:
> “We use 32 units per LSTM layer as a trade-off between representational capacity and computational efficiency.”

This is *exactly* the level of explanation expected—no theory essay, just **coherent reasoning**.

---

#### 3. Directionality (for sequences)
- **Unidirectional**: past → present
- **Bidirectional**: past + future context

Justification:
- time-series forecasting → unidirectional (future unavailable)
- sequence labeling (protein sst8) → bidirectional (full sequence known)

---

### E-3) Regularization & training stability

#### 1. Dropout
- Applied between layers (not across time steps unless explicitly designed)
- Typical values: 0.2–0.5

Justification:
- reduces co-adaptation
- especially important in stacked architectures

In TP 2024–2025, dropout = 0.2 is **required**, so your justification should explain *why dropout exists*, not *why 0.2*.

---

#### 2. Early stopping
- Monitor validation loss
- Stop training when performance degrades

Justification:
- prevents overfitting
- aligns with train/validation/test separation principle

---

#### 3. Batch size
- Small batch → noisy gradients, better generalization
- Large batch → faster but risk of sharp minima

Justification:
> “A moderate batch size is chosen to balance convergence stability and generalization.”

---

### E-4) Loss function & evaluation metrics (must match the task)

This is **non-negotiable** in exams.

#### 1. Classification vs regression
- Regression → MSE / MAE
- Binary classification → binary cross-entropy
- Multiclass classification → categorical cross-entropy

Examples:
- Protein sst8 → categorical cross-entropy (8 classes)
- Bike rental forecasting → MSE or MAE

---

#### 2. Metrics vs loss (they are not the same)
Loss:
- optimized during training

Metrics:
- used to interpret performance

Examples:
- token-level accuracy for sst8
- RMSE for time-series forecasting

The professor explicitly emphasizes **evaluation discipline**, not just training loss.

---

### E-5) Train / validation / test protocol (often forgotten, heavily penalized)

#### 1. Data splitting
- **Time series**: chronological split (no shuffling)
- **Sequences/images**: random split allowed

Justification:
- prevents data leakage
- ensures realistic evaluation

---

#### 2. Qualitative analysis (explicitly required in TP 2022–2023)
Examples:
- compare predicted vs true secondary structures on sample sequences
- visualize predicted vs true curves for bike rentals

Justification:
- helps identify systematic errors
- complements quantitative metrics

---

## 2) Application to the two TPs

## A) TP / Examen 2022–2023  
**Protein secondary structure prediction (sequence → sst8)**

### 1) Data type
- Input `seq`: amino-acid **sequence**.
- Output `sst8`: **sequence of labels** (8 classes), aligned with input.

→ Clearly **sequence data**.

---

### 2) Task output structure
- Multiclass classification **per position**.

---

### 3) Input–output topology
- **Many-to-many** (sequence → sequence).

---

### 4) Appropriate model family

**Option 1 — RNN family (course-consistent default)**  
- **LSTM or GRU**, often **Bidirectional LSTM**.
- Motivation:  
  - decisions depend on context in the sequence;  
  - LSTM handles longer dependencies than vanilla RNN;  
  - bidirectionality captures left and right context.

**Option 2 — 1D CNN over sequences**  
- Captures local motifs (k-mers).
- Still consistent with convolution exploiting local structure (in 1D).

**Option 3 — Transformer encoder**  
- Attention captures long-range dependencies.
- Justifiable if allowed and explained.

---

### 5) How to justify it in the TP notebook

A correct justification (aligned with the course):

- Data are **sequential**, order matters → sequence model required.
- Output is **label per timestep** → many-to-many architecture.
- Use `return_sequences=True`-style outputs.
- Restrict sequence length (e.g., 50 or 100) for computational reasons.
- Use **categorical cross-entropy** for 8-class prediction.

---

### 6) Evaluation

- Token-level accuracy.
- Per-class precision/recall/F1 (important due to imbalance).
- Confusion matrix over 8 classes.
- Qualitative analysis: compare predicted vs true label strings and discuss error patterns.

---

## B) TP Noté 2024–2025  
**Bike rental forecasting (time series)**

### 1) Data type
- Bike rentals indexed by time → **time series**.

---

### 2) Task output structure
- Predict number of rentals → **regression**.

---

### 3) Input–output topology
- Typically **many-to-one** (past window → next value),
  or **many-to-many** for multi-step forecasting.

---

### 4) Model family

The TP explicitly requires:
- **LSTM model**
- **4 LSTM layers**
- **32 units per layer**
- **20% dropout between layers**

This matches the course logic:
- Time dependence requires memory.
- LSTM mitigates short-memory limitations of basic RNNs.
- Dropout combats overfitting in deep recurrent stacks.

---

### 5) Proper justification in the notebook

- CNN (2D) is not adapted: no spatial grid structure.
- LSTM is adapted to temporal dependence and forecasting.
- Dropout improves generalization.
- Window-based dataset construction preserves temporal order.

---

### 6) Evaluation

- Regression metrics: MAE, RMSE (optionally MAPE with care).
- Plot predicted vs true curves on validation/test period.
- Residual analysis.
- Time-aware train/validation split (no random shuffling).

Possible improvement proposals:
- Baseline models (persistence, moving average).
- Feature engineering (hour/day/week seasonality).
- Alternative architectures (GRU, 1D CNN, Transformer).

---

## 3) Reusable mapping table (exam-safe)

| Data type | Task | Default architecture |
|---------|------|---------------------|
| Tabular | Classif./Regression | MLP |
| Image | Classification | CNN / ResNet |
| Image | Segmentation | U-Net |
| Image | Detection | YOLO |
| Time series | Forecasting | LSTM / GRU |
| Text / tokens | Seq / generation | RNN / LSTM / Transformer |
| Sequence labeling | Per-token classes | BiLSTM / Transformer |

---

## 4) Two common mistakes

### Mistake 1 — Choosing by popularity
Using CNNs “because they work well” on non-spatial data.

**Fix**: always justify by *data structure + task structure*.

---

### Mistake 2 — Misaligned outputs or evaluation
- Using sequence → single output when the task is per-token.
- Shuffling time series randomly.

**Fix**: explicitly state topology (many-to-one vs many-to-many) and respect temporal order.

---

**Rule to remember for the exam:**  
> *First justify the data structure, then the task structure, then the architecture family. Hyperparameters come last.*
