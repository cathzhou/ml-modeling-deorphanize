# GPCR-Ligand Binding Predictor

A machine learning pipeline and web application to predict ligand binding interactions with Class A and Class B G-protein Coupled Receptors (GPCRs), leveraging AlphaFold-predicted binary contact maps, receptor-ligand co-expression, ligand N- and C-terminal contact profiles, spatial distance metrics, and family-level annotations. Built to support exploratory receptor-ligand discovery and prioritization.

---

## Project Structure

```
gpcr-binding-predictor/
├── data/                       # Contact maps, metadata, and expression scores
├── model_training/            # Neural network training scripts
├── notebooks/                 # EDA and visualization
├── backend/                   # Inference API for the web app (FastAPI)
├── frontend/                  # Vercel-hosted Next.js frontend
├── diagrams/                  # Architecture diagrams
├── feature_selection_all/     # Initial feature selection results
└── README.md                  # Project documentation
```

---

## Problem

We aim to predict whether a secreted ligand binds a given GPCR using structural and contextual data:

* **AlphaFold Modeled Structures**: 5 models per receptor-ligand pair
* **Binary Contact Maps**: 0 = no contact, 1 = contact for each BW residue
* **Ligand Contact Maps**: Contact profile across ligand N- and C-terminal residues
* **Distance Metrics**: Spatial distances between ligand termini and receptor transmembrane domains
* **Metadata**: Class/family/subfamily (GPCRdb), ligand metadata, AlphaFold metrics
* **Labels**:
    * `known_pair = 1`: Experimentally validated binder
    * `known_pair = 0`: Unknown — potentially unvalidated binder

---

## Input Features for Training

Each input is a receptor-ligand model.

### 1. Residue Contact Vector

* Shape: `[L x 1]` where `L` = total BW residues across all receptors
* Binary values: 0 (no contact), 1 (contact)

### 2. Positional Embedding

* BW positions (e.g., `6.48`) converted to token indices → embedded vectors
* Shape: `[L x D_pos]`

### 3. Ligand N-/C-Terminus Contact Profile

* Columns like `N1_48`, `C1_47`, etc.
* Each represents a ligand residue and contact status (1 or 0)
* Shape: `[R x 1]` where R = number of terminal residues considered

### 4. Distance Metrics

* Columns like `EC_lig1_mid`, `TM1_EC_lig1_mid`, `mid_lig1_CT`, etc.
* Represent physical distances from receptor domains to ligand termini
* Shape: `[D_dist]` where each value is a scalar distance (e.g., float)
* Normalize (z-score) before feeding into the network
* Provide geometric spatial context around binding orientation

### 5. Receptor Metadata

* One-hot or embedded:

    * `gpcrdb_class`
    * `gpcrdb_family`
    * `gpcrdb_subfamily`

### 6. Ligand Metadata

* Calculated from sequence using Uniprot API (from `p2_id` and `p2_range` columns)
    * Molecular weight
    * pI of first 6 AA of N-terminus
    * pI of first 6 AA of C-terminus
    * Sequence length
    * Number of cysteines
    * Distance of closest cysteine to N-term
    * Distance of closest cysteine to C-term 
* Where ligand binds
    * `lig1_end`
    * `lig1_location`
    * `lig1_location_numE`


### 7. AlphaFold Metrics

* Features:
    * `paeR_mean`, `paeL_mean`, `favorability_mean`, `iptm+ptm`, `pLDDT_all_residues`, 
    `pLDDT_lig1`, `pLDDT_rec`, `pLDDT_lig1_NT`, `pLDDT_lig1_CT`


### 8. Expression Co-Expression Vector

* Computed from receptor and ligand expression matrices (tissue × cell-type nTPM)
* Metrics:
    * Pearson correlation
    * Cosine similarity
    * Jaccard index (binary expression overlap)
    * L2 norm of tissue-level expression difference
    * Overlap count in cell types
    * Count of shared top-10 tissues
* Final shape: `[D_expr]`, e.g., `[6–10]` Summary of all metrics

### 9. Label

* `y = 1` if known binder, `0` otherwise

---

## Model Architecture

The model integrates contact features with receptor/ligand context using multi-head attention and dense layers.

### ⛓ Architecture Flow

```
Residue Contact Vector     ─┐
Positional Embedding       ─┤
         [Multi-head Self-Attention]     ───────────────┐
                                               ↓
                                      [Attention Pooling]
                                               ↓
                                [Latent Representation A]
                                               ↓
   N/C-Terminal Ligand Contacts ─┐
            Co-expression Vector ─┐
             Ligand Distance Data ─┐
          Receptor Metadata       ─┤
          Ligand Metadata         ─┤
          AlphaFold Scores        ─┤
             [Context Projection (Dense Layer)]
                                               ↓
                                [Latent Representation B]
                                               ↓
                         Concatenation → [Final Decoder]
                                               ↓
                                  Sigmoid Output: P(binding)
```

---

### Detailed Modules

#### 1. **Multi-Head Self-Attention (MHSA)**

* Input: `[L x D]` matrix (contact + positional encoding)
* Learns relationships across receptor residues
* Output: `[L x D_attn]` → pooled via mean/attention → `[N x D_attn]`

#### 2. **Context Encoder**

* Input vector:

  ```
  X_context = [
    ligand N-/C-terminal contact profile,
    ligand distance-to-receptor metrics,
    gpcrdb_class/family/subfamily embeddings,
    ligand properties,
    AlphaFold metrics,
    expression co-expression vector
  ]
  ```
* Processed via Dense layers:

  ```python
  context_out = nn.Sequential(
      nn.Linear(D_context_input, D_attn),
      nn.ReLU(),
      nn.Dropout(0.1)
  )(X_context)  # shape: [N, D_attn]
  ```

#### 3. **Latent Fusion**

* Combine attention-pooled features and encoded context:

  ```python
  combined = torch.cat([attention_out, context_out], dim=-1)  # shape: [N, 2 * D_attn]
  ```

#### 4. **Final Decoder**

* One or more dense layers with dropout → sigmoid prediction

---

## Training Pipeline

### Loss

* Binary Cross-Entropy
* Focal loss for imbalance

### Class Imbalance Handling

* \~1,260 positives out of 241,540
* Techniques:
    * Weighted loss
    * Negative subsampling
    * Positive bootstrapping

### Optimizer

* `AdamW`, learning rate `1e-4`

### Evaluation Metrics

* AUROC
* AUPRC
* Precision\@k
* F1-score

### Cross-Validation

* `GroupKFold` split by receptor ID
* Prevents data leakage from multi-seed models

---

## Expression Data Extraction Strategy

For expression analysis with all receptor/ligand genes :

### Step 1: Cache Expression Matrices per Gene

* Use HPA API
* Save each gene’s tissue × cell-type `nTPM` matrix to disk
* Avoid redundant API calls

### Step 2: Compute Co-expression Features per Pair

* From cached matrices:
    * Flatten and correlate (Pearson/cosine)
    * Binarize and compute Jaccard index
    * Calculate overlap and distance metrics

### Step 3: Store as Feature Vector

* Compact `[6–10]` vector per pair
* Use as part of `X_context` in model

### Precompute All Pairwise Expression Vectors

* Save to HDF5 or Feather for fast retrieval
* Indexed by (receptor, ligand)

---

## 🌐 Web App (Next.js + FastAPI + MongoDB)

### Input

* Receptor name (e.g., `GPR25`)

### Features

* Query expression data using HPA API
* Retrieve top-k predicted ligands
* Display co-expression scores and contact hotspots
* Visualization: heatmaps, rank tables, structural overlays

---