# Copy of feature_selection.py modified for contact features only
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
from boruta import BorutaPy
import os
warnings.filterwarnings('ignore')

# Set output directory
output_dir = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/contact_only'
os.makedirs(output_dir, exist_ok=True)

# Load contact-only data
contacts_balanced = pd.read_csv(os.path.join(output_dir, 'contacts_balanced_subset.csv'))
contacts_random = pd.read_csv(os.path.join(output_dir, 'contacts_random_subset.csv'))

# Load original data for labels and groups
filepath = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/bm_update_with_spoc_cleaned.csv'
df = pd.read_csv(filepath)

# Get indices from the balanced and random subsets
balanced_indices = contacts_balanced.index
random_indices = contacts_random.index

# Extract corresponding labels and groups
y_balanced = df.loc[balanced_indices, 'known_pair']
y_random = df.loc[random_indices, 'known_pair']
groups_balanced = df.loc[balanced_indices, 'afpd_dir_name']
groups_random = df.loc[random_indices, 'afpd_dir_name']
gtp_families_balanced = df.loc[balanced_indices, 'gtp: Family name']
gtp_families_random = df.loc[random_indices, 'gtp: Family name']

# Convert labels to numerical values
y_balanced = (y_balanced == 'known').astype(int)
y_random = (y_random == 'known').astype(int)

# Create train/val/test splits for both balanced and random subsets
def create_balanced_splits(X, y, groups, test_size=0.2, val_size=0.2):
    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)
    
    n_groups = len(unique_groups)
    n_test = int(n_groups * test_size)
    n_val = int(n_groups * val_size)
    
    test_groups = unique_groups[:n_test]
    val_groups = unique_groups[n_test:n_test + n_val]
    train_groups = unique_groups[n_test + n_val:]
    
    test_mask = groups.isin(test_groups)
    val_mask = groups.isin(val_groups)
    train_mask = groups.isin(train_groups)
    
    return train_mask, val_mask, test_mask

# Create splits for balanced subset
train_mask_balanced, val_mask_balanced, test_mask_balanced = create_balanced_splits(
    contacts_balanced, y_balanced, groups_balanced
)

# Create splits for random subset
train_mask_random, val_mask_random, test_mask_random = create_balanced_splits(
    contacts_random, y_random, groups_random
)

# Function to run feature selection
def run_feature_selection(X_train, y_train, X_val, y_val, feature_names, n_features=20):
    results = {
        'selected_features': {},
        'feature_importances': pd.DataFrame(index=feature_names),
        'evaluation': {}
    }
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    
    # 1. ANOVA F-value
    f_selector = SelectKBest(f_classif, k=n_features)
    f_selector.fit(X_train_scaled, y_train)
    results['selected_features']['anova'] = feature_names[f_selector.get_support()].tolist()
    results['feature_importances']['anova_score'] = f_selector.scores_
    
    # 2. Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k=n_features)
    mi_selector.fit(X_train_scaled, y_train)
    results['selected_features']['mutual_info'] = feature_names[mi_selector.get_support()].tolist()
    results['feature_importances']['mutual_info_score'] = mi_selector.scores_
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_selector = SelectFromModel(rf, max_features=n_features, prefit=True)
    results['selected_features']['random_forest'] = feature_names[rf_selector.get_support()].tolist()
    results['feature_importances']['random_forest_importance'] = rf.feature_importances_
    
    # 4. L1-based selection
    lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    lr.fit(X_train_scaled, y_train)
    l1_selector = SelectFromModel(lr, max_features=n_features, prefit=True)
    results['selected_features']['l1'] = feature_names[l1_selector.get_support()].tolist()
    results['feature_importances']['l1_coef'] = np.abs(lr.coef_[0])
    
    # 5. RFE
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42),
              n_features_to_select=n_features)
    rfe.fit(X_train_scaled, y_train)
    results['selected_features']['rfe'] = feature_names[rfe.get_support()].tolist()
    results['feature_importances']['rfe_rank'] = rfe.ranking_
    
    # 6. Boruta
    rf_boruta = RandomForestClassifier(n_estimators=100, random_state=42)
    boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=42)
    boruta.fit(X_train_scaled.to_numpy(), y_train.to_numpy())
    
    # Get Boruta selected features
    boruta_support = boruta.support_
    boruta_features = feature_names[boruta_support].tolist()
    if len(boruta_features) > n_features:
        boruta_ranks = boruta.ranking_
        top_features_idx = np.argsort(boruta_ranks)[:n_features]
        results['selected_features']['boruta'] = feature_names[top_features_idx].tolist()
    else:
        results['selected_features']['boruta'] = boruta_features
    results['feature_importances']['boruta_rank'] = boruta.ranking_
    results['feature_importances']['boruta_importance'] = boruta.ranking_
    
    # Evaluate each method
    for method, features in results['selected_features'].items():
        y_pred = rf.fit(X_train_scaled[features], y_train).predict(X_val_scaled[features])
        results['evaluation'][method] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, rf.predict_proba(X_val_scaled[features])[:, 1])
        }
        
        # Add confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        results['evaluation'][method]['specificity'] = tn / (tn + fp)
        results['evaluation'][method]['balanced_acc'] = (results['evaluation'][method]['recall'] + 
                                                       results['evaluation'][method]['specificity']) / 2
    
    return results

# Run feature selection on balanced subset
print("\nRunning feature selection on balanced subset...")
balanced_results = run_feature_selection(
    contacts_balanced[train_mask_balanced],
    y_balanced[train_mask_balanced],
    contacts_balanced[val_mask_balanced],
    y_balanced[val_mask_balanced],
    contacts_balanced.columns
)

# Run feature selection on random subset
print("\nRunning feature selection on random subset...")
random_results = run_feature_selection(
    contacts_random[train_mask_random],
    y_random[train_mask_random],
    contacts_random[val_mask_random],
    y_random[val_mask_random],
    contacts_random.columns
)

# Save results
def save_results(results, prefix, output_dir):
    # Save feature importances
    results['feature_importances'].to_csv(
        os.path.join(output_dir, f'{prefix}_feature_importances.csv')
    )
    
    # Save evaluation results
    pd.DataFrame(results['evaluation']).round(3).to_csv(
        os.path.join(output_dir, f'{prefix}_evaluation.csv')
    )
    
    # Save selected features
    with open(os.path.join(output_dir, f'{prefix}_selected_features.txt'), 'w') as f:
        for method, features in results['selected_features'].items():
            f.write(f"\n{method.upper()} SELECTED FEATURES:\n")
            f.write("-" * 50 + "\n")
            for feature in features:
                f.write(f"{feature}\n")
    
    # Create visualization of normalized feature importance rankings
    rankings_normalized = pd.DataFrame(index=results['feature_importances'].index)
    
    for col in results['feature_importances'].columns:
        if col in ['rfe_rank', 'boruta_rank']:
            max_rank = results['feature_importances'][col].max()
            rankings_normalized[f'{col}_normalized'] = 1 - (results['feature_importances'][col] / max_rank)
        elif col == 'boruta_importance':
            rankings_normalized[f'{col}_normalized'] = 1 - (results['feature_importances'][col] / 
                                                          results['feature_importances'][col].max())
        else:
            min_val = results['feature_importances'][col].min()
            max_val = results['feature_importances'][col].max()
            rankings_normalized[f'{col}_normalized'] = ((results['feature_importances'][col] - min_val) / 
                                                      (max_val - min_val))
    
    rankings_normalized['mean_normalized_rank'] = rankings_normalized.mean(axis=1)
    rankings_normalized = rankings_normalized.sort_values('mean_normalized_rank', ascending=False)
    
    # Save normalized rankings
    rankings_normalized.to_csv(os.path.join(output_dir, f'{prefix}_rankings_normalized.csv'))
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(rankings_normalized.head(30), cmap='YlOrRd', center=0.5, annot=True, fmt='.2f')
    plt.title(f'Normalized Feature Importance Rankings - {prefix} (Top 30 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_feature_importance_heatmap.png'))
    plt.close()
    
    # Create correlation matrix of methods
    method_correlations = rankings_normalized.drop('mean_normalized_rank', axis=1).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(method_correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation between Feature Selection Methods - {prefix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_method_correlations.png'))
    plt.close()

# Save results for both subsets
save_results(balanced_results, 'balanced', output_dir)
save_results(random_results, 'random', output_dir)

print("\nResults saved in:", output_dir) 