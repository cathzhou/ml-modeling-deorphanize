# %%
# Contact feature selection script
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
from boruta import BorutaPy
import os
warnings.filterwarnings('ignore')

# Set output directory
output_dir = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/contact_feature_selection'
os.makedirs(output_dir, exist_ok=True)

# Load data with existing splits
filepath_with_splits = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/bm_update_with_spoc_cleaned_splits.csv'
df_splits = pd.read_csv(filepath_with_splits)
filepath = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/bm_update_with_spoc_cleaned.csv'
df = pd.read_csv(filepath)

# Get contact feature columns (from 1.28_25_CP onwards)
contact_start_col = '1.28_25_CP'
contact_cols = df.columns[df.columns.get_loc(contact_start_col):]

# Extract features and labels
X = df[contact_cols].copy()  # Only use contact features
X = X.fillna(0)
y = (df['known_pair'] == 'known').astype(int)

print("Original data shape:", X.shape)
print("\nNumber of contact features:", len(contact_cols))

# Create masks from existing splits
def create_masks_from_splits(split_column):
    """Create boolean masks from split assignments"""
    train_mask = split_column == 'train'
    val_mask = split_column == 'validation'
    test_mask = split_column == 'test'
    subset_mask = split_column != 'unused'
    return train_mask, val_mask, test_mask, subset_mask

# Get masks for both balanced and random approaches
train_mask_balanced, val_mask_balanced, test_mask_balanced, balanced_mask = create_masks_from_splits(df_splits['balanced_split'])
train_mask_random, val_mask_random, test_mask_random, random_mask = create_masks_from_splits(df_splits['random_split'])

# Create subsets using the masks
X_subset = X[balanced_mask].copy()
y_subset = y[balanced_mask]

X_random_subset = X[random_mask].copy()
y_random_subset = y[random_mask]

# Create train/val/test indices for balanced subset
train_idx_balanced = X_subset[train_mask_balanced[balanced_mask]].index
val_idx_balanced = X_subset[val_mask_balanced[balanced_mask]].index
test_idx_balanced = X_subset[test_mask_balanced[balanced_mask]].index

# Create train/val/test indices for random subset
train_idx_random = X_random_subset[train_mask_random[random_mask]].index
val_idx_random = X_random_subset[val_mask_random[random_mask]].index
test_idx_random = X_random_subset[test_mask_random[random_mask]].index

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

# Preprocess data
print("\nPreprocessing balanced subset data...")
X_subset_processed, balanced_stats = preprocess_data(X_subset, contact_cols)
print("\nPreprocessing random subset data...")
X_random_subset_processed, random_stats = preprocess_data(X_random_subset, contact_cols)

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
    X_subset.loc[train_idx_balanced],
    y_subset.loc[train_idx_balanced],
    X_subset.loc[val_idx_balanced],
    y_subset.loc[val_idx_balanced],
    X_subset.loc[test_idx_balanced],
    y_subset.loc[test_idx_balanced],
    contact_cols
)

print("\nAnalyzing balanced subset results...")
analyze_feature_selection_results(
    balanced_results,
    contact_cols,
    os.path.join(output_dir, 'balanced_results')
)

print("\nRunning feature selection on random subset...")
random_results = run_feature_selection(
    X_random_subset.loc[train_idx_random],
    y_random_subset.loc[train_idx_random],
    X_random_subset.loc[val_idx_random],
    y_random_subset.loc[val_idx_random],
    X_random_subset.loc[test_idx_random],
    y_random_subset.loc[test_idx_random],
    contact_cols
)

print("\nAnalyzing random subset results...")
analyze_feature_selection_results(
    random_results,
    contact_cols,
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
    f.write("Contact Features Used in Analysis\n")
    f.write("=" * 30 + "\n\n")
    for feature in contact_cols:
        f.write(f"- {feature}\n")

print(f"\nFeature descriptions saved to: {feature_info_path}")

# Print split distribution summary
print("\nSplit Distribution Summary:")
print("\nBalanced Splits:")
print(df_splits['balanced_split'].value_counts())
print("\nRandom Splits:")
print(df_splits['random_split'].value_counts())

# %%