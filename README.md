# GPCR-Ligand Binding Predictor

A machine learning pipeline (and future web application) to predict ligand binding interactions with Class A and Class B G-protein Coupled Receptors (GPCRs), leveraging AlphaFold-predicted binary contact maps, receptor-ligand co-expression, ligand N- and C-terminal contact profiles, spatial distance metrics, and family-level annotations. Built to support exploratory receptor-ligand discovery and prioritization.

---

## Problem

We aim to predict whether a secreted ligand binds a given GPCR using structural and contextual data:

* **AlphaFold Modeled Structure Metrics**: 5 models per receptor-ligand pair
* **Binary Contact Maps**: 0 = no contact, 1 = contact for each BW residue
* **Ligand Contact Maps**: Contact profile across ligand N- and C-terminal residues
* **Distance Metrics**: Spatial distances between ligand termini and receptor transmembrane domains
* **Metadata**: Class/family/subfamily
* **Coexpression Metrics**: HPA coexpression metrics for the receptor/ligand genes in tissues and cells
* **Ligand Information**: Info about ligand, such as cysteines and pI
* **Labels**:
    * `known_pair = 1`: Experimentally validated binder
    * `known_pair = 0`: Unknown — potentially unvalidated binder

---
## Current Updates:

* Developed a modular data processing pipeline driven by a configurable model file, allowing dynamic specification of input filters (e.g., extracellular-only residues, SPOC-derived metrics), sampling strategy (e.g., random, UMAP-cluster-based), and customizable input/output paths
* Trained a preliminary model using only BW contact data (binary 0/1 contact indicators), with two datasets, one with random sampling and another with UMAP-based sampling. The data was split into training, validation, and test sets with stratified sampling to preserve equal ratio of known vs. unknown
* Binary input representation resembles sequential token data (e.g., text), making it well-suited for self-attention architectures
* Can emphasize global contextual relationships between residue-ligand contacts
* Provide interpretability through attention weights
* Achieved ~70% accuracy, which is in line with expectations given the limited dataset size.
* Generated detailed visual outputs (in model_training folder) including attention maps for 50 test samples, feature importance rankings, confusion matrices, and other visualizations to better understand model behavior

Next steps:
* Conduct cross validation training with the same known set but varying unknown samples to reduce bias and improve generalization
* Scale up to use most of larger dataset (~200k samples)
* Integrate contextual information, such as alphafold metrics and gene expression data, to test whether multi-modal features boost predictive accuracy.

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
* Provide geometric spatial context around binding orientation

### 5. Receptor Metadata

* One-hot or embedded:
    * `ecb: Class or type`
    * `gtp: Family name`
    * `gpcrdb: receptor_class`
    * `gpcrdb: receptor_family`
    * `gpcrdb: subfamily`

### 6. Ligand Information

* Calculated from `sequence` column, ligand sequence comes after the colon
    * `molecular_weight`
    * `n_term_pi`
    * `c_term_pi`
    * `length`
    * `cysteine_count`
    * `dist_cys_to_nterm`
    * `dist_cys_to_cterm`
* Where ligand binds
    * `lig1_end`
    * `lig1_location`
    * `lig1_location_numE`


### 7. AlphaFold Metrics

* Features:
    * `paeR_mean`, `paeL_mean`, `favorability_mean`, `iptm+ptm`, `pLDDT_all_residues`, 
    `pLDDT_lig1`, `pLDDT_rec`, `pLDDT_lig1_NT`, `pLDDT_lig1_CT`


### 8. Expression Co-Expression Data

* Full expression matrices and summary metrics:
    * Tissue expression matrix: `[N_tissues x 2]` raw nTPM values for (receptor, ligand)
    * Cell-type expression matrix: `[N_celltypes x 2]` raw nTPM values for (receptor, ligand)
    * Summary metrics:
        * Pearson correlation
        * Cosine similarity
        * Jaccard index (binary expression overlap)
        * L2 norm of tissue-level expression difference
        * Overlap count in cell types
* Shape: `([N_tissues x 2], [N_celltypes x 2], [D_expr])` where `D_expr = 6`

### 9. CladeOScope Score (not fully integrated yet)

* Single evolutionary coupling score based on phylogenetic profiles
* Computed using min_rank method with COMB5 clade combination
* Lower scores indicate stronger evolutionary co-dependence
* Shape: `[1]` scalar value per receptor-ligand pair

### 10. Label

* `y = 1` if known binder, `0` otherwise (from `known_pair` column )

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
     Tissue Expression Matrix    ─┐
   Cell-type Expression Matrix   ─┐
      Co-expression Metrics      ─┐
         CladeOScope Score       ─┐
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

## Configuration and Data Preprocessing

### Configuration Files

The project uses two main configuration files to manage data processing and model training:

#### Data Configuration File
- **Purpose**: Defines column mappings and feature groupings for the input data
- **Location**: `model_training/configs/data_config.json`
- **Key Components**:
  - Column groupings for different feature types (residue contacts, distance metrics, ligand contacts, etc.)
  - Categorical column definitions for encoding
  - Expression feature column mappings
  - Metadata column specifications

#### Model Configuration File
- **Purpose**: Controls model architecture, training parameters, and feature selection
- **Location**: `model_training/configs/model_config.json`
- **Key Components**:
  - Model architecture parameters (embedding dimensions, attention heads, etc.)
  - Training hyperparameters (learning rate, batch size, epochs)
  - Feature group activation settings
  - Data splitting and balancing parameters
  - Model name and save paths

### Data Preprocessing Pipeline

The data preprocessing system handles the complex input data structure and prepares it for the MHSA model:

#### Feature Processing
- **Residue Contacts**: Binary contact maps (0/1) indicating residue-residue interactions
- **Distance Metrics**: Spatial distance measurements between ligand termini and receptor domains
- **Ligand Contact Profiles**: Contact patterns at ligand N- and C-termini
- **AlphaFold Metrics**: Quality scores from AlphaFold structure predictions
- **Expression Features**: Co-expression data from HPA database
- **Metadata**: Categorical information about receptor families and ligand properties

#### Normalization and Encoding
- **Z-score normalization** for continuous features
- **Min-max scaling** to 0-1 range for certain feature groups
- **Label encoding** for categorical variables
- **Feature-specific processing** based on data type (binary features remain unchanged)

#### Data Splitting and Balancing
- **Group-aware splitting**: Ensures all models from the same receptor-ligand pair stay in the same split
- **Balanced sampling**: Maintains equal representation of known and unknown pairs
- **Cluster-based selection**: Uses UMAP clustering for spatially balanced unknown pair selection
- **Multiple rounds**: Supports training with different unknown pair subsets

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
    tissue_expression_matrix,      # [N_tissues x 2]
    celltype_expression_matrix,    # [N_celltypes x 2] 
    co_expression_metrics,         # [D_expr]
    cladescope_score              # [1]
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
* Save each gene's tissue × cell-type `nTPM` matrix to disk
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

* Save to csv
* Indexed by (receptor, ligand)

---

### Multi-Head Self-Attention (MHSA) Model

The core model uses a transformer-inspired architecture specifically designed for residue contact prediction:

#### Model Architecture
- **Input Projection**: Projects each feature to a high-dimensional embedding space
- **Positional Encoding**: Adds sequence position information to maintain spatial context
- **Multi-Head Attention**: 8 attention heads that learn different types of feature relationships
- **Residual Connections**: Helps with gradient flow and training stability
- **Layer Normalization**: Stabilizes training and improves convergence
- **Feed-Forward Networks**: Non-linear transformations between attention layers
- **Global Pooling**: Aggregates sequence-level features into a single prediction

#### Attention Mechanism
- **Self-Attention**: Each feature attends to all other features to capture global dependencies
- **Scaled Dot-Product**: Standard attention computation with scaling for numerical stability
- **Multi-Head Processing**: Parallel attention heads capture different types of relationships
- **Attention Weights**: Provide interpretable insights into feature importance

#### Training Features
- **Binary Cross-Entropy Loss**: Standard loss function for binary classification
- **AdamW Optimizer**: Adaptive learning rate with weight decay
- **Dropout Regularization**: Prevents overfitting during training
- **Early Stopping**: Monitors validation loss to prevent overfitting
- **Model Checkpointing**: Saves best model based on validation performance

### Input Data Structure

The model accepts preprocessed feature vectors with the following characteristics:

#### Feature Dimensions
- **Residue Contacts**: Variable length binary vectors (typically 200-300 features)
- **Distance Metrics**: Fixed-length continuous vectors (normalized distances)
- **Ligand Contacts**: Binary contact profiles at ligand termini
- **AlphaFold Scores**: Quality metrics from structure prediction
- **Expression Data**: Co-expression correlation and similarity metrics
- **Metadata**: Encoded categorical information

#### Data Format
- **CSV Input**: Tabular data with columns for each feature type
- **Code Identifiers**: Unique identifiers for each receptor-ligand pair
- **Split Labels**: Train/validation/test assignments
- **Target Labels**: Binary binding predictions (known/unknown pairs)

#### Preprocessing Output
- **Normalized Features**: All features scaled to appropriate ranges
- **Encoded Categories**: Categorical variables converted to numerical representations
- **Balanced Splits**: Equal representation of positive and negative samples
- **Groups**: Related samples kept together in splits

### Visualization and Analysis Tools

The project includes comprehensive tools for model interpretation and analysis:

#### Attention Visualization
- **Individual Code Analysis**: Generate attention heatmaps for specific receptor-ligand pairs
- **Multi-Head Visualization**: Display attention patterns across all 8 attention heads
- **Feature Labeling**: Show feature names on axes for interpretability
- **Batch Processing**: Generate plots for all test codes automatically

#### Model Performance Analysis
- **Confidence Matrix**: Visualize prediction confidence vs true labels
- **ROC and PR Curves**: Standard classification performance metrics
- **Confusion Matrix**: Detailed classification results
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and AUC

#### Feature Importance Analysis
- **Attention-Based Importance**: Use attention weights to identify important features
- **Top Feature Ranking**: Rank features by their attention scores
- **Attention Heatmaps**: Visualize attention patterns for top features
- **Cross-Sample Averaging**: Aggregate importance across multiple samples

#### Usage Examples
```bash
# Generate attention plots for all test codes
python visualize_attention.py --all-test-codes

# Limit to first 10 test codes
python visualize_attention.py --all-test-codes --max-codes 10

# Skip performance analysis, only generate attention plots
python visualize_attention.py --all-test-codes --skip-performance --skip-feature-importance
```

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
