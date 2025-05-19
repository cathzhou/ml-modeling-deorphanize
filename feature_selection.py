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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# %%
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

print("Data shape:", X.shape)
print("\nNumber of features:", len(numerical_cols))
print("\nFeature names:", numerical_cols)

# %%
# Split data while preserving GTP family representation
unique_groups = np.unique(groups)

# Create dictionary of groups by GTP family and known/unknown status
family_groups = {}
for family in gtp_families.unique():
    family_mask = gtp_families == family
    family_groups[family] = {
        'known': [g for g in unique_groups if g in groups[family_mask & (y == 1)].unique()],
        'unknown': [g for g in unique_groups if g in groups[family_mask & (y == 0)].unique()]
    }

# Print initial distribution
print("\nInitial distribution of groups by GTP family:")
for family, status_dict in family_groups.items():
    print(f"\n{family}:")
    print(f"Known groups: {len(status_dict['known'])}")
    print(f"Unknown groups: {len(status_dict['unknown'])}")

# %%
# Function to split groups while maintaining family representation
def split_groups_by_family(family_groups, test_size=0.2, val_size=0.2):
    train_groups, val_groups, test_groups = [], [], []
    
    for family, status_dict in family_groups.items():
        for status in ['known', 'unknown']:
            groups_list = status_dict[status]
            if not groups_list:
                continue
                
            np.random.shuffle(groups_list)
            n_groups = len(groups_list)
            
            # Calculate split sizes
            n_test = max(1, int(n_groups * test_size))  # At least 1 group in test
            n_val = max(1, int(n_groups * val_size))    # At least 1 group in val
            n_train = n_groups - n_test - n_val         # Rest in train
            
            # If not enough groups, adjust splits
            if n_train == 0:
                n_train = 1
                n_test = min(1, n_groups - 1)
                n_val = max(0, n_groups - n_train - n_test)
            
            # Split the groups
            test_groups.extend(groups_list[:n_test])
            val_groups.extend(groups_list[n_test:n_test + n_val])
            train_groups.extend(groups_list[n_test + n_val:])
    
    return train_groups, val_groups, test_groups

# Perform the split
train_groups, val_groups, test_groups = split_groups_by_family(family_groups)

# Create masks
test_mask = groups.isin(test_groups)
val_mask = groups.isin(val_groups)
train_mask = groups.isin(train_groups)

# Add split information to the original dataframe
df['split'] = 'train'  # Default to train
df.loc[val_mask, 'split'] = 'validation'
df.loc[test_mask, 'split'] = 'test'

# Export the dataframe with split information
output_path = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/bm_update_with_spoc_cleaned_with_splits.csv'
df.to_csv(output_path, index=False)
print(f"\nDataframe with split information saved to: {output_path}")

# Print distribution summary by GTP family and split
print("\nDistribution summary by GTP family and split:")
print("-" * 50)
for family in gtp_families.unique():
    print(f"\nGTP Family: {family}")
    family_mask = gtp_families == family
    for split in ['train', 'validation', 'test']:
        split_mask = df['split'] == split
        known_count = sum((y == 1) & family_mask & split_mask)
        total_count = sum(family_mask & split_mask)
        print(f"{split:12}: {known_count} known out of {total_count} total")

# Split data
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("\nFinal set sizes:")
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# %%
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# %%
# Apply feature selection methods and get feature importances
n_features = 20  # Number of features to select
selected_features = {}
feature_importances = pd.DataFrame(index=X_train.columns)

# 1. ANOVA F-value
f_selector = SelectKBest(f_classif, k=n_features)
f_selector.fit(X_train_scaled, y_train)
selected_features['anova'] = X_train.columns[f_selector.get_support()].tolist()
# Get ANOVA F-scores
feature_importances['anova_score'] = f_selector.scores_

# 2. Mutual Information
mi_selector = SelectKBest(mutual_info_classif, k=n_features)
mi_selector.fit(X_train_scaled, y_train)
selected_features['mutual_info'] = X_train.columns[mi_selector.get_support()].tolist()
# Get Mutual Information scores
feature_importances['mutual_info_score'] = mi_selector.scores_

# 3. Random Forest Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_selector = SelectFromModel(rf, max_features=n_features, prefit=True)
selected_features['random_forest'] = X_train.columns[rf_selector.get_support()].tolist()
# Get Random Forest feature importances
feature_importances['random_forest_importance'] = rf.feature_importances_

# 4. L1-based selection (Logistic Regression)
lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
lr.fit(X_train_scaled, y_train)
l1_selector = SelectFromModel(lr, max_features=n_features, prefit=True)
selected_features['l1'] = X_train.columns[l1_selector.get_support()].tolist()
# Get L1 coefficients (absolute values)
feature_importances['l1_coef'] = np.abs(lr.coef_[0])

# 5. Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42),
          n_features_to_select=n_features)
rfe.fit(X_train_scaled, y_train)
selected_features['rfe'] = X_train.columns[rfe.get_support()].tolist()
# Get RFE ranking (1 is selected, higher numbers are worse)
feature_importances['rfe_rank'] = rfe.ranking_

# %%
# Normalize importance scores to 0-1 scale for comparison
for col in feature_importances.columns:
    if col != 'rfe_rank':  # Don't normalize RFE ranks
        feature_importances[col] = (feature_importances[col] - feature_importances[col].min()) / \
                                 (feature_importances[col].max() - feature_importances[col].min())

# Create rankings for each method
rankings = pd.DataFrame(index=X_train.columns)
for col in feature_importances.columns:
    if col == 'rfe_rank':
        # For RFE, lower rank is better
        rankings[f'{col}_rank'] = feature_importances[col].rank()
    else:
        # For others, higher score is better
        rankings[f'{col}_rank'] = feature_importances[col].rank(ascending=False)

# Calculate mean rank across all methods
rankings['mean_rank'] = rankings.mean(axis=1)
rankings = rankings.sort_values('mean_rank')

# %%
# Save feature importance results
importance_path = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/feature_importances.csv'
feature_importances.to_csv(importance_path)
print(f"\nFeature importances saved to: {importance_path}")

ranking_path = '/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/feature_rankings.csv'
rankings.to_csv(ranking_path)
print(f"Feature rankings saved to: {ranking_path}")

# %%
# Print top features by mean rank
print("\nTop 20 features by mean rank across all methods:")
print("-" * 50)
print(rankings.head(20))

# %%
# Create a heatmap of feature importance rankings
plt.figure(figsize=(12, 8))
sns.heatmap(rankings.iloc[:20, :-1], cmap='YlOrRd_r', annot=True, fmt='.0f')
plt.title('Feature Importance Rankings Across Methods (Top 20 Features)')
plt.tight_layout()
plt.savefig('/Users/catherinez/Research/Conserved_N-term/ml_modeling/data/feature_importance_heatmap.png')
plt.show()

# %%
# Print features selected by multiple methods
feature_counts = pd.Series([feature for features in selected_features.values() for feature in features])
feature_counts = feature_counts.value_counts()

print("\nFeatures selected by multiple methods:")
print("-" * 50)
print("Number of methods | Features")
print("-" * 50)
for n_methods in range(len(selected_features), 0, -1):
    features = feature_counts[feature_counts == n_methods].index.tolist()
    if features:
        print(f"{n_methods:^16} | {', '.join(features)}")

# %%
# Create visualization of feature importance
feature_importance = pd.DataFrame(index=X_train.columns)

# For each method, mark selected features
for method, features in selected_features.items():
    feature_importance[method] = 0
    feature_importance.loc[features, method] = 1

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importance, cmap='YlOrRd', cbar=False)
plt.title('Feature Selection Results Across Methods')
plt.tight_layout()
plt.savefig('feature_selection_results.png')
plt.show()

# %%
# Save selected features to file
with open('selected_features.txt', 'w') as f:
    for method, features in selected_features.items():
        f.write(f"\n{method.upper()} SELECTED FEATURES:\n")
        f.write("-" * 50 + "\n")
        for feature in features:
            f.write(f"{feature}\n")

# %%
# Optional: Display common features across methods
common_features = set.intersection(*[set(features) for features in selected_features.values()])
print("\nFeatures selected by all methods:")
print("-" * 50)
for feature in sorted(common_features):
    print(feature)

# %%
# Optional: Display feature importance from Random Forest
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by Random Forest importance:")
print("-" * 50)
print(rf_importance.head(20))

# %%
