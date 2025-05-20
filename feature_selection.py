# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
from boruta import BorutaPy
import os
warnings.filterwarnings('ignore')

# Load and preprocess data
filepath = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/bm_update_with_spoc_cleaned.csv'
df = pd.read_csv(filepath)

# Extract target and group labels
y = df['known_pair']
groups = df['afpd_dir_name']
gtp_families = df['gtp: Family name']

# Convert labels to numerical values (known=1, unknown=0)
y = (y == 'known').astype(int)

# Select numerical columns only, excluding target, group columns, and specific columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
exclude_cols = ['known_pair', 'model_num', 'pred_num']
numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

# Fill NA values with 0
X = df[numerical_cols].fillna(0)

print("Original data shape:", X.shape)
print("\nNumber of features:", len(numerical_cols))

def get_balanced_subset(df, gtp_families, y, groups, target_ratio=0.5):
    """
    Create balanced subset with specified known/unknown ratio, including all known pairs
    target_ratio: desired proportion of known samples (e.g., 0.5 means 50% known, 50% unknown)
    """
    # First, get all known pairs
    known_mask = y == 1
    all_known_indices = df[known_mask].index
    
    # Calculate how many unknown samples we need for the target ratio
    n_known = len(all_known_indices)
    n_unknown_needed = int(n_known * (1 - target_ratio) / target_ratio)
    
    # Get unknown samples by family to maintain family distribution
    unknown_by_family = {}
    for family in gtp_families.unique():
        family_mask = gtp_families == family
        unknown_indices = df[family_mask & (y == 0)].index
        if len(unknown_indices) > 0:
            unknown_by_family[family] = unknown_indices
    
    # Calculate number of unknown samples needed per family
    total_unknown_samples = sum(len(indices) for indices in unknown_by_family.values())
    unknown_indices = []
    
    for family, indices in unknown_by_family.items():
        # Calculate proportion of unknown samples to take from this family
        family_proportion = len(indices) / total_unknown_samples
        n_family_unknown = int(n_unknown_needed * family_proportion)
        
        # Get unique groups for unknown samples in this family
        family_unknown_groups = groups[indices].unique()
        
        if len(family_unknown_groups) > 0:
            # Select groups to achieve target numbers
            n_groups = max(1, min(len(family_unknown_groups), 
                          n_family_unknown // (len(indices) // len(family_unknown_groups))))
            selected_groups = np.random.choice(family_unknown_groups, 
                                            size=int(n_groups), 
                                            replace=False)
            
            # Get all samples from selected groups
            family_unknown_subset = df[groups.isin(selected_groups) & (y == 0)].index
            
            # If we got too many samples, randomly subsample
            if len(family_unknown_subset) > n_family_unknown:
                family_unknown_subset = np.random.choice(family_unknown_subset, 
                                                      size=n_family_unknown, 
                                                      replace=False)
            
            unknown_indices.extend(family_unknown_subset)
    
    # Combine all indices
    subset_indices = list(all_known_indices) + list(unknown_indices)
    return subset_indices

def get_random_balanced_subset(df, y, groups, target_ratio=0.5):
    """
    Create balanced subset with specified known/unknown ratio, randomly selecting samples
    target_ratio: desired proportion of known samples (e.g., 0.5 means 50% known, 50% unknown)
    """
    # Get all known pairs
    known_mask = y == 1
    all_known_indices = df[known_mask].index
    
    # Calculate how many unknown samples we need
    n_known = len(all_known_indices)
    n_unknown_needed = int(n_known * (1 - target_ratio) / target_ratio)
    
    # Get all unknown pairs
    unknown_mask = y == 0
    all_unknown_indices = df[unknown_mask].index
    
    # Get unique groups for unknown samples
    unknown_groups = groups[all_unknown_indices].unique()
    
    # Randomly select unknown groups
    n_groups_needed = n_unknown_needed // (len(all_unknown_indices) // len(unknown_groups))
    selected_unknown_groups = np.random.choice(unknown_groups, 
                                             size=int(n_groups_needed), 
                                             replace=False)
    
    # Get samples from selected groups
    unknown_subset = df[groups.isin(selected_unknown_groups) & (y == 0)].index
    
    # If we got too many samples, randomly subsample
    if len(unknown_subset) > n_unknown_needed:
        unknown_subset = np.random.choice(unknown_subset, 
                                        size=n_unknown_needed, 
                                        replace=False)
    
    # Combine indices
    subset_indices = list(all_known_indices) + list(unknown_subset)
    return subset_indices

# Create balanced subset with 50-50 split
print("\nCreating balanced subset of data...")
np.random.seed(42)
subset_indices = get_balanced_subset(df, gtp_families, y, groups, target_ratio=0.5)

print("\nCreating random balanced subset...")
random_subset_indices = get_random_balanced_subset(df, y, groups, target_ratio=0.5)

# Create subset dataframes
df_subset = df.loc[subset_indices]
X_subset = X.loc[subset_indices]
y_subset = y[subset_indices]
groups_subset = groups[subset_indices]
gtp_families_subset = gtp_families[subset_indices]

df_random_subset = df.loc[random_subset_indices]
X_random_subset = X.loc[random_subset_indices]
y_random_subset = y[random_subset_indices]
groups_random_subset = groups[random_subset_indices]
gtp_families_random_subset = gtp_families[random_subset_indices]

def create_balanced_splits(groups, y, test_size=0.2, val_size=0.2):
    """
    Create balanced splits while maintaining group integrity.
    
    Args:
        groups: Series containing group labels
        y: Series containing target labels
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
    
    Returns:
        train_mask, val_mask, test_mask: Boolean arrays indicating split membership
    """
    unique_groups = np.unique(groups)
    
    # Separate groups by known/unknown status
    known_groups = []
    unknown_groups = []
    
    for g in unique_groups:
        group_mask = groups == g
        if np.any(y[group_mask] == 1):
            known_groups.append(g)
        else:
            unknown_groups.append(g)
    
    # Convert to numpy arrays
    known_groups = np.array(known_groups)
    unknown_groups = np.array(unknown_groups)
    
    # Shuffle groups
    np.random.shuffle(known_groups)
    np.random.shuffle(unknown_groups)
    
    # Calculate split sizes for known groups
    n_known = len(known_groups)
    n_known_test = int(n_known * test_size)
    n_known_val = int(n_known * val_size)
    
    # Calculate split sizes for unknown groups
    n_unknown = len(unknown_groups)
    n_unknown_test = int(n_unknown * test_size)
    n_unknown_val = int(n_unknown * val_size)
    
    # Split groups
    known_test = known_groups[:n_known_test]
    known_val = known_groups[n_known_test:n_known_test + n_known_val]
    known_train = known_groups[n_known_test + n_known_val:]
    
    unknown_test = unknown_groups[:n_unknown_test]
    unknown_val = unknown_groups[n_unknown_test:n_unknown_test + n_unknown_val]
    unknown_train = unknown_groups[n_unknown_test + n_unknown_val:]
    
    # Combine splits
    test_groups = np.concatenate([known_test, unknown_test])
    val_groups = np.concatenate([known_val, unknown_val])
    train_groups = np.concatenate([known_train, unknown_train])
    
    # Create boolean masks
    test_mask = groups.isin(test_groups)
    val_mask = groups.isin(val_groups)
    train_mask = groups.isin(train_groups)
    
    return train_mask, val_mask, test_mask


# Create train/val/test splits for both subsets
print("\nCreating splits for balanced subset...")
train_mask_balanced, val_mask_balanced, test_mask_balanced = create_balanced_splits(
    groups_subset, y_subset
)

print("\nCreating splits for random subset...")
train_mask_random, val_mask_random, test_mask_random = create_balanced_splits(
    groups_random_subset, y_random_subset
)

def write_distribution_info(df_data, y_data, groups_data, gtp_families_data, train_mask, val_mask, test_mask, filename):
    """
    Write distribution information about the dataset to a file.
    
    Args:
        df_data: DataFrame containing all data
        y_data: Series containing target labels
        groups_data: Series containing group labels
        gtp_families_data: Series containing family labels
        train_mask: Boolean array for training set
        val_mask: Boolean array for validation set
        test_mask: Boolean array for test set
        filename: Path to output file
    """
    with open(filename, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("1. Overall Data Shape\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples: {len(df_data)}\n")
        f.write(f"Features: {df_data.shape[1]}\n\n")
        
        # Overall class distribution
        f.write("2. Overall Class Distribution\n")
        f.write("-" * 20 + "\n")
        class_dist = pd.Series(y_data).value_counts()
        class_dist_norm = pd.Series(y_data).value_counts(normalize=True)
        f.write("Absolute counts:\n")
        f.write(f"Known pairs: {class_dist.get(1, 0)}\n")
        f.write(f"Unknown pairs: {class_dist.get(0, 0)}\n")
        f.write("\nPercentages:\n")
        f.write(f"Known pairs: {class_dist_norm.get(1, 0):.2%}\n")
        f.write(f"Unknown pairs: {class_dist_norm.get(0, 0):.2%}\n\n")
        
        # Split statistics
        f.write("3. Split Statistics\n")
        f.write("-" * 20 + "\n")
        splits = {
            'Train': train_mask,
            'Validation': val_mask,
            'Test': test_mask
        }
        
        for split_name, mask in splits.items():
            # Ensure mask is boolean
            if not isinstance(mask, (np.ndarray, pd.Series)):
                continue
                
            f.write(f"\n{split_name} Set:\n")
            f.write(f"Total samples: {mask.sum()}\n")
            
            # Get data for this split
            split_y = y_data[mask]
            split_class_dist = pd.Series(split_y).value_counts()
            split_class_norm = pd.Series(split_y).value_counts(normalize=True)
            
            f.write(f"Known pairs: {split_class_dist.get(1, 0)} ({split_class_norm.get(1, 0):.2%})\n")
            f.write(f"Unknown pairs: {split_class_dist.get(0, 0)} ({split_class_norm.get(0, 0):.2%})\n")
            
            split_groups = groups_data[mask]
            split_families = gtp_families_data[mask]
            f.write(f"Unique groups: {len(split_groups.unique())}\n")
            f.write(f"Unique families: {len(split_families.unique())}\n")
            
            # Family distribution within split
            f.write("\nFamily distribution:\n")
            split_family_dist = pd.crosstab(split_families, split_y)
            f.write(str(split_family_dist))
            f.write("\n\nFamily percentages:\n")
            split_family_pct = split_family_dist.div(split_family_dist.sum(axis=1), axis=0) * 100
            f.write(str(split_family_pct.round(3)))
            f.write("\n")
        
        f.write("\n4. Distribution by GTP Family\n")
        f.write("-" * 20 + "\n")
        family_dist = pd.crosstab(gtp_families_data, y_data)
        f.write("Counts per family:\n")
        f.write(str(family_dist))
        f.write("\n\nPercentages per family:\n")
        family_pct = family_dist.div(family_dist.sum(axis=1), axis=0) * 100
        f.write(str(family_pct.round(3)))
        f.write("\n\n")
        
        f.write("5. Detailed Family Statistics\n")
        f.write("-" * 20 + "\n")
        for family in sorted(gtp_families_data.unique()):
            family_mask = gtp_families_data == family
            f.write(f"\nFamily: {family}\n")
            f.write(f"Total samples: {family_mask.sum()}\n")
            family_y = y_data[family_mask]
            f.write(f"Known pairs: {(family_y == 1).sum()}\n")
            f.write(f"Unknown pairs: {(family_y == 0).sum()}\n")
            f.write(f"Known/Unknown ratio: {(family_y == 1).mean():.2%}\n")
            
            # Split distribution for this family
            f.write("\nSplit distribution:\n")
            for split_name, split_mask in splits.items():
                if not isinstance(split_mask, (np.ndarray, pd.Series)):
                    continue
                    
                combined_mask = family_mask & split_mask
                if combined_mask.sum() > 0:
                    f.write(f"{split_name}:\n")
                    f.write(f"  Samples: {combined_mask.sum()}\n")
                    split_family_y = y_data[combined_mask]
                    f.write(f"  Known: {(split_family_y == 1).sum()}\n")
                    f.write(f"  Unknown: {(split_family_y == 0).sum()}\n")
                    split_family_groups = groups_data[combined_mask]
                    f.write(f"  Groups: {len(split_family_groups.unique())}\n")
            f.write("\n")
        
        f.write("\n6. Group Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total unique groups: {len(groups_data.unique())}\n")
        avg_samples_per_group = len(df_data) / len(groups_data.unique())
        f.write(f"Average samples per group: {avg_samples_per_group:.2f}\n\n")
        
        # Group statistics by split
        f.write("Group statistics by split:\n")
        for split_name, split_mask in splits.items():
            if not isinstance(split_mask, (np.ndarray, pd.Series)):
                continue
                
            split_groups = groups_data[split_mask].unique()
            f.write(f"\n{split_name}:\n")
            f.write(f"  Unique groups: {len(split_groups)}\n")
            f.write(f"  Average samples per group: {split_mask.sum() / len(split_groups):.2f}\n")
            
            # Check group overlap
            other_splits = {k: v for k, v in splits.items() if k != split_name}
            for other_name, other_mask in other_splits.items():
                if not isinstance(other_mask, (np.ndarray, pd.Series)):
                    continue
                    
                other_groups = groups_data[other_mask].unique()
                overlap = len(set(split_groups) & set(other_groups))
                f.write(f"  Group overlap with {other_name}: {overlap} groups\n")

# Create balanced splits
print("\nCreating splits for balanced subset...")
train_mask_balanced, val_mask_balanced, test_mask_balanced = create_balanced_splits(
    groups_subset, y_subset
)

print("\nCreating splits for random subset...")
train_mask_random, val_mask_random, test_mask_random = create_balanced_splits(
    groups_random_subset, y_random_subset
)

# Save distribution information to text files
print("\nSaving distribution information...")
write_distribution_info(
    df_subset, y_subset, groups_subset, gtp_families_subset,
    train_mask_balanced, val_mask_balanced, test_mask_balanced,
    '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/distribution_balanced.txt'
)

write_distribution_info(
    df_random_subset, y_random_subset, groups_random_subset, gtp_families_random_subset,
    train_mask_random, val_mask_random, test_mask_random,
    '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/distribution_random.txt'
)

print("\nDetailed distribution information saved to:")
print("- distribution_balanced.txt")
print("- distribution_random.txt")

# Create balanced and random subsets
print("\nCreating balanced subset...")
X_subset = df.loc[subset_indices]
X_random_subset = df.loc[random_subset_indices]

# Save subsets with index
output_dir = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/feature_selection'
os.makedirs(output_dir, exist_ok=True)

X_subset.to_csv(os.path.join(output_dir, 'balanced_subset.csv'))
X_random_subset.to_csv(os.path.join(output_dir, 'random_subset.csv'))

print("\nSubsets saved to:", output_dir)

# Select only numerical features for analysis
def get_numerical_features(df):
    """
    Get numerical features from DataFrame, excluding specific columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of numerical column names
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude specific columns
    exclude_cols = ['known_pair', 'model_num', 'pred_num']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    return numerical_cols

def preprocess_data(X, feature_names):
    """
    Preprocess data by handling missing values and scaling.
    
    Args:
        X: Input features DataFrame
        feature_names: List of feature names to use
    
    Returns:
        Preprocessed DataFrame with selected features
    """
    # Select features
    X_selected = X[feature_names].copy()
    
    # Replace infinite values with NaN
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    
    # Get statistics for each column
    column_stats = pd.DataFrame({
        'missing_count': X_selected.isnull().sum(),
        'missing_percent': (X_selected.isnull().sum() / len(X_selected) * 100).round(2),
        'infinite_count': ((X_selected == np.inf) | (X_selected == -np.inf)).sum(),
        'mean': X_selected.mean(),
        'median': X_selected.median(),
        'std': X_selected.std()
    })
    
    # Fill NaN values with median for each column
    X_selected = X_selected.fillna(X_selected.median())
    
    return X_selected, column_stats

# Get numerical features
numerical_features = get_numerical_features(X_subset)
print(f"\nNumber of numerical features selected: {len(numerical_features)}")

# Preprocess data
print("\nPreprocessing balanced subset data...")
X_subset_processed, balanced_stats = preprocess_data(X_subset, numerical_features)
print("\nPreprocessing random subset data...")
X_random_subset_processed, random_stats = preprocess_data(X_random_subset, numerical_features)

# Save preprocessing statistics
stats_dir = os.path.join(output_dir, 'preprocessing_stats')
os.makedirs(stats_dir, exist_ok=True)

balanced_stats.to_csv(os.path.join(stats_dir, 'balanced_subset_stats.csv'))
random_stats.to_csv(os.path.join(stats_dir, 'random_subset_stats.csv'))

# Create preprocessing report
with open(os.path.join(stats_dir, 'preprocessing_report.txt'), 'w') as f:
    f.write("Data Preprocessing Report\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Balanced Subset Statistics\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total samples: {len(X_subset)}\n")
    f.write(f"Features with missing values: {(balanced_stats['missing_count'] > 0).sum()}\n")
    f.write(f"Features with infinite values: {(balanced_stats['infinite_count'] > 0).sum()}\n\n")
    
    f.write("Random Subset Statistics\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total samples: {len(X_random_subset)}\n")
    f.write(f"Features with missing values: {(random_stats['missing_count'] > 0).sum()}\n")
    f.write(f"Features with infinite values: {(random_stats['infinite_count'] > 0).sum()}\n\n")
    
    # List features with high missing value percentages
    threshold = 5  # 5% threshold for reporting
    f.write("\nFeatures with high missing value percentages (>5%):\n")
    f.write("-" * 50 + "\n")
    f.write("\nBalanced Subset:\n")
    high_missing = balanced_stats[balanced_stats['missing_percent'] > threshold]
    for idx, row in high_missing.iterrows():
        f.write(f"{idx}: {row['missing_percent']:.2f}% missing\n")
    
    f.write("\nRandom Subset:\n")
    high_missing = random_stats[random_stats['missing_percent'] > threshold]
    for idx, row in high_missing.iterrows():
        f.write(f"{idx}: {row['missing_percent']:.2f}% missing\n")

print(f"\nPreprocessing statistics saved to: {stats_dir}")

# Scale the features for the subset
def run_feature_selection(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, n_features=20):
    """
    Run multiple feature selection methods and evaluate their performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        feature_names: Names of the features
        n_features: Number of features to select (default: 20)
    
    Returns:
        Dictionary containing results from all feature selection methods
    """
    results = {}
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame with feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Dictionary to store feature selection methods
    selection_methods = {
        'anova': {
            'method': SelectKBest(f_classif, k=n_features),
            'needs_scaling': True
        },
        'mutual_info': {
            'method': SelectKBest(mutual_info_classif, k=n_features),
            'needs_scaling': False
        },
        'random_forest': {
            'method': RandomForestClassifier(n_estimators=100, random_state=42),
            'selector': SelectFromModel,
            'needs_scaling': False
        },
        'l1_logistic': {
            'method': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            'selector': SelectFromModel,
            'needs_scaling': True
        },
        'rfe': {
            'method': RFE(
                estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                n_features_to_select=n_features
            ),
            'needs_scaling': False
        }
    }
    
    # Run each feature selection method
    for method_name, method_info in selection_methods.items():
        print(f"Running {method_name} selection...")
        method = method_info['method']
        
        # Fit the method
        if 'selector' in method_info:
            method.fit(X_train_scaled, y_train)
            selector = method_info['selector'](method, max_features=n_features, prefit=True)
            support = selector.get_support()
            if hasattr(method, 'feature_importances_'):
                scores = method.feature_importances_
            else:
                scores = np.abs(method.coef_[0]) if method_name != 'elasticnet' else np.abs(method.coef_)
        else:
            method.fit(X_train_scaled, y_train)
            support = method.get_support()
            scores = method.scores_ if hasattr(method, 'scores_') else method.ranking_
        
        # Get selected feature names using boolean indexing
        selected_features = np.array(feature_names)[support].tolist()
        
        results[method_name] = {
            'selected_features': selected_features,
            'scores': scores,
            'support': support
        }
    
    # Run Boruta
    print("Running Boruta...")
    rf_boruta = RandomForestClassifier(n_estimators=100, random_state=42)
    boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=42)
    
    # Convert to numpy arrays for Boruta
    X_train_np = X_train_scaled.to_numpy()
    y_train_np = y_train.to_numpy()
    
    # Fit Boruta
    boruta.fit(X_train_np, y_train_np)
    
    # Get Boruta results
    boruta_support = boruta.support_
    boruta_features = np.array(feature_names)[boruta_support].tolist()
    
    # If more than n_features selected, take top n_features by ranking
    if len(boruta_features) > n_features:
        boruta_ranks = boruta.ranking_
        top_features_idx = np.argsort(boruta_ranks)[:n_features]
        boruta_features = np.array(feature_names)[top_features_idx].tolist()
        boruta_support = np.zeros_like(boruta_support, dtype=bool)
        boruta_support[top_features_idx] = True
    
    results['boruta'] = {
        'selected_features': boruta_features,
        'scores': boruta.ranking_,
        'support': boruta_support
    }
    
    # Calculate correlation-based features
    print("Running correlation analysis...")
    corr_with_target = pd.DataFrame()
    corr_with_target['correlation'] = [np.corrcoef(X_train_scaled.iloc[:, i], y_train)[0, 1] 
                                     for i in range(X_train_scaled.shape[1])]
    corr_with_target.index = feature_names
    
    # Select top features by absolute correlation
    top_corr_features = corr_with_target['correlation'].abs().nlargest(n_features).index
    corr_support = np.isin(feature_names, top_corr_features)
    
    results['correlation'] = {
        'selected_features': top_corr_features.tolist(),
        'scores': corr_with_target['correlation'].abs().values,
        'support': corr_support
    }
    
    # Evaluate each method on both validation and test sets
    for method_name, method_results in results.items():
        selected_features = method_results['selected_features']
        
        # Train and evaluate a Random Forest classifier using selected features
        rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_eval.fit(X_train_scaled[selected_features], y_train)
        
        # Validation set metrics
        y_val_pred = rf_eval.predict(X_val_scaled[selected_features])
        y_val_proba = rf_eval.predict_proba(X_val_scaled[selected_features])[:, 1]
        
        # Test set metrics
        y_test_pred = rf_eval.predict(X_test_scaled[selected_features])
        y_test_proba = rf_eval.predict_proba(X_test_scaled[selected_features])[:, 1]
        
        # Calculate metrics for both sets
        method_results['metrics'] = {
            'validation': {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'roc_auc': roc_auc_score(y_val, y_val_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba)
            }
        }
    
    return results
def analyze_feature_selection_results(results, feature_names, output_dir):
    """
    Analyze and save feature selection results.
    
    Args:
        results: Dictionary containing results from all feature selection methods
        feature_names: List of all feature names
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each method's results
    for method_name, method_results in results.items():
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        with open(os.path.join(method_dir, 'analysis.txt'), 'w') as f:
            f.write(f"Analysis for {method_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Selected features
            f.write("Selected features:\n")
            for feature in method_results['selected_features']:
                f.write(f"- {feature}\n")
            f.write("\n")
            
            # Performance metrics for both validation and test sets
            f.write("Performance metrics:\n")
            for split in ['validation', 'test']:
                f.write(f"\n{split.capitalize()} Set:\n")
                for metric, value in method_results['metrics'][split].items():
                    f.write(f"- {metric}: {value}\n")
            f.write("\n")
            
            # Feature importance scores if available
            if 'scores' in method_results:
                f.write("\nFeature importance scores:\n")
                scores = method_results['scores']
                if len(scores) == len(feature_names):
                    for feature, score in zip(feature_names, scores):
                        f.write(f"- {feature}: {score:.3f}\n")
                else:
                    f.write("(Scores not available in compatible format)\n")
        
        # Create feature importance plot if scores are available
        if 'scores' in method_results and len(method_results['scores']) == len(feature_names):
            plt.figure(figsize=(12, 6))
            feature_importance = pd.Series(method_results['scores'], index=feature_names)
            feature_importance.sort_values(ascending=True).plot(kind='barh')
            plt.title(f'Feature Importance Scores - {method_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(method_dir, 'feature_importance.png'))
            plt.close()
    
    # Feature consistency analysis
    selected_features_matrix = np.zeros((len(feature_names), len(results)))
    for i, (method_name, method_results) in enumerate(results.items()):
        selected_features = method_results['selected_features']
        selected_features_matrix[:, i] = [1 if f in selected_features else 0 for f in feature_names]
    
    # Calculate feature selection frequency
    selection_frequency = pd.DataFrame(
        selected_features_matrix,
        index=feature_names,
        columns=results.keys()
    )
    selection_frequency['total'] = selection_frequency.sum(axis=1)
    selection_frequency = selection_frequency.sort_values('total', ascending=False)
    
    # Save feature selection frequency
    selection_frequency.to_csv(os.path.join(output_dir, 'feature_selection_frequency.csv'))
    
    # Create feature selection frequency plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(selection_frequency.iloc[:, :-1], cmap='YlOrRd', cbar_kws={'label': 'Selected'})
    plt.title('Feature Selection Frequency Across Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_selection_frequency.png'))
    plt.close()
    
    # Create summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Feature Selection Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Most consistently selected features
        f.write("Most consistently selected features:\n")
        consistent_features = selection_frequency[selection_frequency['total'] > len(results)/2]
        for feature, row in consistent_features.iterrows():
            f.write(f"- {feature} (selected by {int(row['total'])} methods)\n")
        f.write("\n")
        
        # Performance comparison across methods
        f.write("Performance comparison across methods:\n")
        for split in ['validation', 'test']:
            f.write(f"\n{split.capitalize()} Set:\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                f.write(f"\n{metric}:\n")
                for method_name, method_results in results.items():
                    value = method_results['metrics'][split][metric]
                    f.write(f"- {method_name}: {value:.3f}\n")
        
        # Best performing method for each metric
        f.write("\nBest performing methods:\n")
        for split in ['validation', 'test']:
            f.write(f"\n{split.capitalize()} Set:\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                best_method = max(results.items(), 
                                key=lambda x: x[1]['metrics'][split][metric])
                best_value = best_method[1]['metrics'][split][metric]
                f.write(f"- Best {metric}: {best_method[0]} ({best_value:.3f})\n")

    return selection_frequency

# Run feature selection on both subsets
print("\nRunning feature selection on balanced subset...")
balanced_results = run_feature_selection(
    X_subset_processed[train_mask_balanced],
    y_subset[train_mask_balanced],
    X_subset_processed[val_mask_balanced],
    y_subset[val_mask_balanced],
    X_subset_processed[test_mask_balanced],
    y_subset[test_mask_balanced],
    numerical_features
)

print("\nAnalyzing balanced subset results...")
analyze_feature_selection_results(
    balanced_results,
    numerical_features,
    os.path.join(output_dir, 'balanced_results')
)

print("\nRunning feature selection on random subset...")
random_results = run_feature_selection(
    X_random_subset_processed[train_mask_random],
    y_random_subset[train_mask_random],
    X_random_subset_processed[val_mask_random],
    y_random_subset[val_mask_random],
    X_random_subset_processed[test_mask_random],
    y_random_subset[test_mask_random],
    numerical_features
)

print("\nAnalyzing random subset results...")
analyze_feature_selection_results(
    random_results,
    numerical_features,
    os.path.join(output_dir, 'random_results')
)

print("\nFeature selection analysis completed. Results saved in:", output_dir)

# Compare results between balanced and random subsets
print("\nComparing balanced and random subset results...")
comparison_dir = os.path.join(output_dir, 'comparison')
os.makedirs(comparison_dir, exist_ok=True)

def compare_feature_selection_results(balanced_results, random_results, output_dir):
    """
    Compare feature selection results between balanced and random subsets.
    
    Args:
        balanced_results: Results from balanced subset
        random_results: Results from random subset
        output_dir: Directory to save comparison results
    """
    # Compare selected features
    with open(os.path.join(output_dir, 'feature_comparison.txt'), 'w') as f:
        f.write("Feature Selection Comparison: Balanced vs Random Subsets\n")
        f.write("=" * 60 + "\n\n")
        
        for method in balanced_results.keys():
            f.write(f"\n{method.upper()}\n")
            f.write("-" * 20 + "\n")
            
            balanced_features = set(balanced_results[method]['selected_features'])
            random_features = set(random_results[method]['selected_features'])
            
            common_features = balanced_features & random_features
            balanced_only = balanced_features - random_features
            random_only = random_features - balanced_features
            
            f.write(f"Common features ({len(common_features)}):\n")
            for feature in sorted(common_features):
                f.write(f"- {feature}\n")
            
            f.write(f"\nFeatures unique to balanced subset ({len(balanced_only)}):\n")
            for feature in sorted(balanced_only):
                f.write(f"- {feature}\n")
            
            f.write(f"\nFeatures unique to random subset ({len(random_only)}):\n")
            for feature in sorted(random_only):
                f.write(f"- {feature}\n")
            
            # Calculate Jaccard similarity
            jaccard = len(common_features) / len(balanced_features | random_features)
            f.write(f"\nJaccard similarity: {jaccard:.3f}\n")
    
    # Compare performance metrics for both validation and test sets
    for split in ['validation', 'test']:
        comparison_metrics = pd.DataFrame()
        for method in balanced_results.keys():
            balanced_metrics = pd.Series(balanced_results[method]['metrics'][split], name='balanced')
            random_metrics = pd.Series(random_results[method]['metrics'][split], name='random')
            diff = balanced_metrics - random_metrics
            
            method_comparison = pd.concat([balanced_metrics, random_metrics, diff], axis=1)
            method_comparison.columns = ['Balanced', 'Random', 'Difference']
            
            comparison_metrics[method] = diff
        
        # Save performance comparison
        comparison_metrics.to_csv(os.path.join(output_dir, f'performance_comparison_{split}.csv'))
        
        # Create visualization of performance differences
        plt.figure(figsize=(12, 6))
        sns.heatmap(comparison_metrics, cmap='RdBu', center=0, annot=True, fmt='.3f')
        plt.title(f'Performance Differences (Balanced - Random) - {split.capitalize()} Set')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_comparison_{split}.png'))
        plt.close()
        
        print(f"\nSummary of performance differences (Balanced - Random) - {split.capitalize()} Set:")
        print(comparison_metrics.round(3))
    
    return comparison_metrics

# Compare results
comparison_metrics = compare_feature_selection_results(
    balanced_results,
    random_results,
    comparison_dir
)

print("\nComparison results saved in:", comparison_dir)

# Save feature names and their descriptions
feature_info_path = os.path.join(output_dir, 'feature_descriptions.txt')
with open(feature_info_path, 'w') as f:
    f.write("Numerical Features Used in Analysis\n")
    f.write("=" * 30 + "\n\n")
    for feature in numerical_features:
        f.write(f"- {feature}\n")

print(f"\nFeature descriptions saved to: {feature_info_path}")


# %%

# Create split assignment columns
def create_split_assignments(df, subset_indices, train_mask, val_mask, test_mask):
    """Create a series with split assignments including unused samples"""
    split_assignments = pd.Series(index=df.index, data='unused')
    subset_df = pd.Series(index=subset_indices)
    
    # Assign splits for samples in the subset
    subset_df[train_mask] = 'train'
    subset_df[val_mask] = 'validation'
    subset_df[test_mask] = 'test'
    
    # Update the full dataset assignments
    split_assignments[subset_indices] = subset_df
    
    return split_assignments

# Create split assignment columns for both approaches
balanced_splits = create_split_assignments(
    df, subset_indices, 
    train_mask_balanced, val_mask_balanced, test_mask_balanced
)
random_splits = create_split_assignments(
    df, random_subset_indices,
    train_mask_random, val_mask_random, test_mask_random
)

# Add split columns to original dataframe
df['balanced_split'] = balanced_splits
df['random_split'] = random_splits

# Save updated dataframe
print("\nSaving updated CSV with split assignments...")
df.to_csv(filepath, index=False)
print("Split assignments saved to original CSV file.")

# Print split distribution summary
print("\nSplit Distribution Summary:")
print("\nBalanced Splits:")
print(df['balanced_split'].value_counts())
print("\nRandom Splits:")
print(df['random_split'].value_counts())

# %%
