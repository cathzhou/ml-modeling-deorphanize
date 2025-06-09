import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GPCRDataset(Dataset):
    """PyTorch Dataset for GPCR-Ligand binding data."""
    
    def __init__(self, 
                 features: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 active_feature_groups: Set[str],
                 is_training: bool = True):
        """
        Args:
            features: Dictionary of feature arrays
            labels: Binary labels (0/1)
            active_feature_groups: Set of feature groups to include
            is_training: Whether this is for training
        """
        self.features = features
        self.labels = torch.FloatTensor(labels)
        self.active_feature_groups = active_feature_groups
        self.is_training = is_training
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Get features based on active feature groups
        item = {}
        
        if 'residue_contacts' in self.active_feature_groups:
            item['residue_contacts'] = torch.FloatTensor(self.features['residue_contacts'][idx])
            item['positional_embedding'] = torch.FloatTensor(self.features['positional_embedding'][idx])
            
        if 'distance_metrics' in self.active_feature_groups:
            item['distance_metrics'] = torch.FloatTensor(self.features['distance_metrics'][idx])
            
        if 'ligand_contact_sum' in self.active_feature_groups:
            item['ligand_contact_sum'] = torch.FloatTensor(self.features['ligand_contact_sum'][idx])
            
        if 'ligand_contact_indiv' in self.active_feature_groups:
            item['ligand_contact_indiv'] = torch.FloatTensor(self.features['ligand_contact_indiv'][idx])
            
        if 'receptor_metadata' in self.active_feature_groups:
            item['receptor_metadata'] = torch.FloatTensor(self.features['receptor_metadata'][idx])
            
        if 'ligand_metadata' in self.active_feature_groups:
            item['ligand_metadata'] = torch.FloatTensor(self.features['ligand_metadata'][idx])
            
        if 'alphafold_metrics' in self.active_feature_groups:
            item['alphafold_metrics'] = torch.FloatTensor(self.features['alphafold_metrics'][idx])
            
        if 'expression_features' in self.active_feature_groups:
            item['expression_features'] = torch.FloatTensor(self.features['expression_features'][idx])
            
        if 'spoc_metrics' in self.active_feature_groups:
            item['spoc_metrics'] = torch.FloatTensor(self.features['spoc_metrics'][idx])
            
        if 'expression_profiles' in self.active_feature_groups:
            item['receptor_expression'] = torch.FloatTensor(self.features['receptor_expression'][idx])
            item['ligand_expression'] = torch.FloatTensor(self.features['ligand_expression'][idx])
        
        return item, self.labels[idx]

class DataPreprocessor:
    """Handles data preprocessing for GPCR-Ligand binding prediction."""
    
    def __init__(self, config_path: str = 'model_training/config.json'):
        """
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.encoders = {}
        self.expression_cache = {}
        self.active_feature_groups = {
            group for group, active in self.config['feature_groups'].items() 
            if active
        }
        logging.info("Active feature groups:")
        for group in sorted(self.active_feature_groups):
            logging.info(f"  - {group}")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _get_column_groups(self) -> Dict[str, List[str]]:
        """Get column groupings from config for active feature groups."""
        column_groups = {}
        
        # Map feature groups to their column configurations
        group_to_config = {
            'residue_contacts': 'residue_contact_columns',
            'distance_metrics': 'distance_metric_columns',
            'ligand_contact_sum': 'ligand_contact_sum_columns',
            'ligand_contact_indiv': 'ligand_contact_indiv_columns',
            'alphafold_metrics': 'alphafold_metric_columns',
            'expression_features': 'expression_feature_columns',
            'ligand_metadata': 'ligand_metadata_columns',
            'spoc_metrics': 'spoc_metric_columns'
        }
        
        # Only include active feature groups
        for group in self.active_feature_groups:
            if group in group_to_config:
                config_key = group_to_config[group]
                if config_key in self.config:
                    column_groups[group] = self.config[config_key]
                else:
                    logging.warning(f"Config key '{config_key}' not found for feature group '{group}'")
        
        return column_groups

    def _load_expression_profile(self, uniprot_id: str, data_dir: str = 'data/expression_cache') -> np.ndarray:
        """
        Load expression profile for a given UniProt ID.
        
        Args:
            uniprot_id: UniProt ID of the protein
            data_dir: Directory containing expression profile CSVs
            
        Returns:
            Expression profile as numpy array
        """
        if uniprot_id in self.expression_cache:
            return self.expression_cache[uniprot_id]
            
        file_path = os.path.join(data_dir, f"{uniprot_id}_expression.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Expression profile not found for {uniprot_id}")
            # Return zeros with same shape as other profiles
            if len(self.expression_cache) > 0:
                # Use first cached profile's shape
                first_profile = next(iter(self.expression_cache.values()))
                return np.zeros_like(first_profile)
            else:
                # If no profiles loaded yet, need to load one to get shape
                example_file = next(f for f in os.listdir(data_dir) if f.endswith('_expression.csv'))
                example_profile = pd.read_csv(os.path.join(data_dir, example_file), index_col=0).values.flatten()
                return np.zeros_like(example_profile)
        
        try:
            profile_df = pd.read_csv(file_path, index_col=0)
            profile = profile_df.values.flatten()
            self.expression_cache[uniprot_id] = profile
            return profile
        except Exception as e:
            logging.error(f"Error loading expression profile for {uniprot_id}: {e}")
            return np.zeros_like(next(iter(self.expression_cache.values())) if self.expression_cache else None)

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Union[str, List[str], Dict[str, bool]]]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.
        
        Args:
            df: Input DataFrame
            filters: Dictionary of filter conditions with format:
                {
                    'column_equals': {'column_name': 'value'},  # Exact match
                    'column_in': {'column_name': ['value1', 'value2']},  # Match any in list
                    'non_nan_columns': ['column1', 'column2'],  # Must have non-NaN values
                    'all_non_nan_groups': ['group1', 'group2']  # All columns in group must be non-NaN
                }
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        initial_len = len(filtered_df)
        
        if filters:
            # Apply exact match filters
            if 'column_equals' in filters:
                for col, value in filters['column_equals'].items():
                    filtered_df = filtered_df[filtered_df[col] == value]
            
            # Apply list match filters
            if 'column_in' in filters:
                for col, values in filters['column_in'].items():
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
            
            # Apply non-NaN filters for specific columns
            if 'non_nan_columns' in filters:
                for col in filters['non_nan_columns']:
                    filtered_df = filtered_df[~filtered_df[col].isna()]
            
            # Apply non-NaN filters for entire feature groups
            if 'all_non_nan_groups' in filters:
                column_groups = self._get_column_groups()
                for group in filters['all_non_nan_groups']:
                    if group in column_groups:
                        group_cols = column_groups[group]
                        filtered_df = filtered_df[~filtered_df[group_cols].isna().any(axis=1)]
            
            # Log filtering results
            final_len = len(filtered_df)
            removed = initial_len - final_len
            logging.info(f"Filtering removed {removed} samples ({removed/initial_len*100:.1f}% of data)")
            logging.info(f"Remaining samples: {final_len}")
        
        return filtered_df
    
    def balance_dataset(self, df: pd.DataFrame, split_method: str = 'umap', random_state: int = 42) -> pd.DataFrame:
        """
        Balance the dataset by downsampling the majority class (unknown pairs).
        For UMAP method, selects unknown pairs that are well-distributed in UMAP space.
        For random method, uses random selection.
        
        Args:
            df: Input DataFrame
            split_method: Method for selecting unknown pairs ('umap' or 'random')
            random_state: Random seed
        """
        known_pairs = df[df['known_pair'] == 1]
        unknown_pairs = df[df['known_pair'] == 0]
        print(unknown_pairs)
        
        # Get the number of samples to keep
        n_known = len(known_pairs)
        
        if split_method == 'umap':
            # Use UMAP coordinates to select well-distributed unknown pairs
            umap_coords = unknown_pairs[['nmfUMAP1_af_qc', 'nmfUMAP2_af_qc']].values
            print(umap_coords)
            
            # Use KMeans to select centroids that are well-distributed
            from sklearn.cluster import KMeans
            n_clusters = n_known  # One cluster per known pair
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(umap_coords)
            
            # Select the point closest to each centroid
            from sklearn.metrics.pairwise import euclidean_distances
            selected_indices = []
            for i in range(n_clusters):
                cluster_points = unknown_pairs[cluster_labels == i]
                if len(cluster_points) > 0:
                    cluster_coords = cluster_points[['nmfUMAP1_af_qc', 'nmfUMAP2_af_qc']].values
                    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
                    distances = euclidean_distances(cluster_coords, centroid)
                    closest_idx = distances.argmin()
                    selected_indices.append(cluster_points.iloc[closest_idx].name)
            
            # If we don't have enough points, add random points from larger clusters
            if len(selected_indices) < n_known:
                remaining = n_known - len(selected_indices)
                cluster_sizes = pd.Series(cluster_labels).value_counts()
                large_clusters = cluster_sizes[cluster_sizes > 1].index
                additional_points = []
                for cluster_id in large_clusters:
                    cluster_points = unknown_pairs[cluster_labels == cluster_id]
                    if len(additional_points) < remaining:
                        # Exclude already selected points
                        available_points = cluster_points[~cluster_points.index.isin(selected_indices)]
                        if len(available_points) > 0:
                            additional_points.append(available_points.iloc[0].name)
                selected_indices.extend(additional_points[:remaining])
            
            unknown_balanced = unknown_pairs.loc[selected_indices]
            
        else:  # random method
            # Randomly sample from unknown pairs
            np.random.seed(random_state)
            unknown_balanced = unknown_pairs.sample(n=n_known, random_state=random_state)
        
        # Combine and shuffle
        balanced_df = pd.concat([known_pairs, unknown_balanced])
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Convert to binary values
        balanced_df['known_pair'] = (balanced_df['known_pair'] == 'known').astype(int)
        
        logging.info(f"Balanced dataset: {len(balanced_df)} total pairs")
        logging.info(f"Known pairs: {(balanced_df['known_pair'] == 1).sum()}")
        logging.info(f"Unknown pairs (after balancing): {(balanced_df['known_pair'] == 0).sum()}")
        
        if split_method == 'umap':
            logging.info("Used UMAP-based selection for unknown pairs")
        else:
            logging.info("Used random selection for unknown pairs")
        
        return balanced_df
    
    def preprocess_data(self, 
                       data_path: str,
                       split_method: str = 'umap',
                       test_size: float = 0.1,
                       valid_size: float = 0.1,
                       random_state: int = 42,
                       feature_groups: Optional[Dict[str, bool]] = None,
                       filters: Optional[Dict[str, Union[str, List[str], Dict[str, bool]]]] = None) -> Tuple[Dict[str, Dataset], Dict[str, StandardScaler]]:
        """
        Preprocess data and create train/valid/test splits.
        
        Args:
            data_path: Path to input data CSV
            split_method: Method for splitting unknown pairs ('umap', 'random')
            test_size: Fraction of data for test set
            valid_size: Fraction of data for validation set
            random_state: Random seed
            feature_groups: Optional dict to override feature group settings from config
            filters: Optional dict of filter conditions with format:
                {
                    'column_equals': {'column_name': 'value'},  # Exact match
                    'column_in': {'column_name': ['value1', 'value2']},  # Match any in list
                    'non_nan_columns': ['column1', 'column2'],  # Must have non-NaN values
                    'all_non_nan_groups': ['group1', 'group2']  # All columns in group must be non-NaN
                }
            
        Returns:
            datasets: Dict of train/valid/test datasets
            scalers: Dict of fitted scalers for each feature group
        """
        # Update active feature groups if provided
        if feature_groups is not None:
            self.active_feature_groups = {
                group for group, active in feature_groups.items() 
                if active
            }
            logging.info("Updated active feature groups:")
            for group in sorted(self.active_feature_groups):
                logging.info(f"  - {group}")
        
        logging.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert known_pair to binary values
        if 'known_pair' in df.columns:
            # Convert string 'known'/'unknown' to 1/0
            if df['known_pair'].dtype == 'object':
                df['known_pair'] = (df['known_pair'] == 'known').astype(int)
            print(df['known_pair'])
            logging.info(f"Converted known_pair to binary values (0/1)")
            logging.info(f"Known pairs (1): {(df['known_pair'] == 1).sum()}")
            logging.info(f"Unknown pairs (0): {(df['known_pair'] == 0).sum()}")
        
        # Apply filters if provided
        if filters:
            logging.info("Applying data filters...")
            df = self._apply_filters(df, filters)
        
        # Balance the dataset
        df = self.balance_dataset(df, split_method=split_method, random_state=random_state)
        
        # Split data
        if split_method == 'umap':
            train_idx, valid_idx, test_idx = self._split_by_umap(
                df, 
                test_size=test_size,
                valid_size=valid_size,
                random_state=random_state
            )
        else:
            train_idx, valid_idx, test_idx = self._split_random(
                df,
                test_size=test_size,
                valid_size=valid_size,
                random_state=random_state
            )
            
        # Add split labels to dataframe
        df['split'] = 'train'  # default
        df.loc[valid_idx, 'split'] = 'valid'
        df.loc[test_idx, 'split'] = 'test'
        
        # Save processed dataset with split labels
        output_path = Path(data_path).parent / 'processed_features_with_splits.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed dataset with split labels to {output_path}")
        
        # Create splits
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        test_df = df.iloc[test_idx]
        
        # Log split statistics
        logging.info("\nSplit Statistics:")
        for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
            n_known = (split_df['known_pair'] == 1).sum()
            n_unknown = (split_df['known_pair'] == 0).sum()
            logging.info(f"{split_name} set:")
            logging.info(f"  Total pairs: {len(split_df)}")
            logging.info(f"  Known pairs: {n_known}")
            logging.info(f"  Unknown pairs: {n_unknown}")
            logging.info(f"  Known/Unknown ratio: {n_known/n_unknown:.2f}")
        
        # Process features for each split
        train_features, train_labels = self._process_split(train_df, is_training=True)
        valid_features, valid_labels = self._process_split(valid_df, is_training=False)
        test_features, test_labels = self._process_split(test_df, is_training=False)
        
        # Create datasets
        datasets = {
            'train': GPCRDataset(train_features, train_labels, self.active_feature_groups, is_training=True),
            'valid': GPCRDataset(valid_features, valid_labels, self.active_feature_groups, is_training=False),
            'test': GPCRDataset(test_features, test_labels, self.active_feature_groups, is_training=False)
        }
        
        return datasets, self.scalers
    
    def _split_by_umap(self, 
                      df: pd.DataFrame,
                      test_size: float,
                      valid_size: float,
                      random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data using UMAP coordinates to ensure equidistant distribution in each split.
        Uses KMeans clustering on UMAP coordinates to create well-distributed splits.
        """
        # Get UMAP coordinates for all data
        umap_coords = df[['nmfUMAP1_af_qc', 'nmfUMAP2_af_qc']].values
        
        # Calculate number of clusters for each split
        total_clusters = int(1/min(test_size, valid_size))  # Ensure enough clusters for smallest split
        n_test = int(total_clusters * test_size)
        n_valid = int(total_clusters * valid_size)
        
        # Use KMeans to create well-distributed clusters
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=total_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(umap_coords)
        
        # Assign clusters to splits while maintaining class balance within each cluster
        unique_clusters = np.unique(cluster_labels)
        np.random.seed(random_state)
        
        # Shuffle clusters
        np.random.shuffle(unique_clusters)
        
        # Assign clusters to splits
        test_clusters = unique_clusters[:n_test]
        valid_clusters = unique_clusters[n_test:n_test + n_valid]
        train_clusters = unique_clusters[n_test + n_valid:]
        
        # Get indices for each split
        test_idx = df[np.isin(cluster_labels, test_clusters)].index
        valid_idx = df[np.isin(cluster_labels, valid_clusters)].index
        train_idx = df[np.isin(cluster_labels, train_clusters)].index
        
        # Log split sizes
        logging.info("\nSplit sizes after UMAP-based splitting:")
        logging.info(f"Train: {len(train_idx)} samples")
        logging.info(f"Valid: {len(valid_idx)} samples")
        logging.info(f"Test: {len(test_idx)} samples")
        
        return train_idx, valid_idx, test_idx
    
    def _split_random(self,
                     df: pd.DataFrame,
                     test_size: float,
                     valid_size: float,
                     random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data randomly while maintaining class balance."""
        # First split off test set
        train_valid_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            stratify=df['known_pair']
        )
        
        # Then split remaining data into train and validation
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size/(1-test_size),
            random_state=random_state,
            stratify=df.loc[train_valid_idx, 'known_pair']
        )
        
        return train_idx, valid_idx, test_idx
    
    def _process_split(self,
                      df: pd.DataFrame,
                      is_training: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Process features for a data split."""
        features = {}
        column_groups = self._get_column_groups()
        
        # Process each feature group
        for group_name, columns in column_groups.items():
            if group_name in self.active_feature_groups:
                if is_training:
                    scaler = StandardScaler()
                    features[group_name] = scaler.fit_transform(df[columns])
                    self.scalers[group_name] = scaler
                else:
                    features[group_name] = self.scalers[group_name].transform(df[columns])
        
        # Process categorical features
        if 'receptor_metadata' in self.active_feature_groups:
            for cat_col in self.config['categorical_columns']:
                if is_training:
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(df[cat_col])
                    self.encoders[cat_col] = encoder
                else:
                    encoded = self.encoders[cat_col].transform(df[cat_col])
                features[f'{cat_col}_encoded'] = encoded
        
        # Process expression profiles if enabled
        if 'expression_profiles' in self.active_feature_groups:
            # Load and process receptor expression profiles
            receptor_profiles = np.stack([
                self._load_expression_profile(uniprot_id) 
                for uniprot_id in df['p1_id']
            ])
            
            # Load and process ligand expression profiles
            ligand_profiles = np.stack([
                self._load_expression_profile(uniprot_id)
                for uniprot_id in df['p2_id']
            ])
            
            if is_training:
                # Fit scalers on training data
                receptor_scaler = StandardScaler()
                ligand_scaler = StandardScaler()
                
                features['receptor_expression'] = receptor_scaler.fit_transform(receptor_profiles)
                features['ligand_expression'] = ligand_scaler.fit_transform(ligand_profiles)
                
                self.scalers['receptor_expression'] = receptor_scaler
                self.scalers['ligand_expression'] = ligand_scaler
            else:
                # Transform using fitted scalers
                features['receptor_expression'] = self.scalers['receptor_expression'].transform(receptor_profiles)
                features['ligand_expression'] = self.scalers['ligand_expression'].transform(ligand_profiles)
        
        # Get labels
        labels = df['known_pair'].values
        
        return features, labels

def create_data_loaders(datasets: Dict[str, Dataset],
                       batch_size: int,
                       num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders."""
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    return loaders 