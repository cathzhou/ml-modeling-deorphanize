import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import json
import os
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GPCRDataset(Dataset):
    """PyTorch Dataset for GPCR-Ligand binding prediction."""
    
    def __init__(self, 
                 features: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 active_feature_groups: Set[str],
                 config: Dict,
                 is_training: bool = True):
        """Initialize dataset.
        
        Args:
            features: Dictionary of feature arrays
            labels: Array of labels
            active_feature_groups: Set of active feature groups
            config: Model configuration dictionary
            is_training: Whether this is training data
        """
        self.features = features
        self.labels = labels
        self.active_feature_groups = active_feature_groups
        self.config = config
        self.is_training = is_training
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get features based on active feature groups
        feature_dict = {}
        
        # Get standard feature groups
        for group in self.active_feature_groups:
            feature_dict[group] = torch.FloatTensor(self.features[group][idx])
        
        # Get individual categorical features
        for cat_col in self.config['categorical_columns']:
            feature_dict[f'{cat_col}_encoded'] = torch.LongTensor([self.features[f'{cat_col}_encoded'][idx]])
        
        # Get label
        label = torch.FloatTensor([self.labels[idx]])
        
        return feature_dict, label

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
            'residue_contacts': 'residue_contact_columns', # no normalization,0 and 1
            'distance_metrics': 'distance_metric_columns', # Z score normalized, 0 to 1
            'ligand_contact_sum': 'ligand_contact_sum_columns', # no normalization, 0 to 1
            'ligand_contact_indiv': 'ligand_contact_indiv_columns', # no normalization, 0 to 1
            'alphafold_metrics': 'alphafold_metric_columns', # Z score normalized, 0 to 1
            'expression_features': 'expression_feature_columns', # take absolute value, Z score normalized, 0 to 1
            'ligand_metadata': 'ligand_metadata_columns', # Z score normalized, 0 to 1
            'spoc_metrics': 'spoc_metric_columns' # Z score normalized, 0 to 1
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
        print("Known pairs: ", n_known)
        print("Unknown pairs: ", len(unknown_pairs))
        
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
                
        logging.info(f"Balanced dataset: {len(balanced_df)} total pairs")
        logging.info(f"Known pairs: {(balanced_df['known_pair'] == 1).sum()}")
        logging.info(f"Unknown pairs (after balancing): {(balanced_df['known_pair'] == 0).sum()}")
        
        if split_method == 'umap':
            logging.info("Used UMAP-based selection for unknown pairs")
        else:
            logging.info("Used random selection for unknown pairs")
        
        return balanced_df
    
    def _save_processed_features(self, features: Dict[str, np.ndarray], df: pd.DataFrame, split: str, output_dir: str):
        """Save processed features to CSV files.
        
        Args:
            features: Dictionary of processed features
            df: Original dataframe
            split: Split name (train/valid/test)
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each feature group separately
        for group_name, feature_array in features.items():
            if group_name.endswith('_encoded'):
                # For encoded categorical features, save both encoded and original values
                cat_col = group_name.replace('_encoded', '')
                encoded_df = pd.DataFrame({
                    'original_value': df[cat_col],
                    'encoded_value': feature_array,
                    'cleaned_value': df[cat_col].apply(self._clean_text)
                })
                encoded_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}.csv'), index=False)
            else:
                # For normalized features, get original column names from config
                if group_name in self._get_column_groups():
                    columns = self._get_column_groups()[group_name]
                    feature_df = pd.DataFrame(feature_array, columns=columns)
                    feature_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}_normalized.csv'), index=False)
                else:
                    # For other features (like expression profiles)
                    feature_df = pd.DataFrame(feature_array)
                    feature_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}.csv'), index=False)

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
            if df['known_pair'].dtype == 'object':
                df['known_pair'] = (df['known_pair'] == 'known').astype(int)
            logging.info(f"Converted known_pair to binary values (0/1)")
            logging.info(f"Known pairs (1): {(df['known_pair'] == 1).sum()}")
            logging.info(f"Unknown pairs (0): {(df['known_pair'] == 0).sum()}")
        
        # Apply filters if provided
        if filters:
            logging.info("Applying data filters...")
            df = self._apply_filters(df, filters)
        
        # Pre-process categorical features before splitting
        for cat_col in self.config['categorical_columns']:
            # Clean text
            df[f'{cat_col}_cleaned'] = df[cat_col].apply(self._clean_text)
            
            # Encode categorical values
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(df[f'{cat_col}_cleaned'])
            self.encoders[cat_col] = encoder
            
            # Apply Z-score normalization and 0-1 scaling
            encoded_reshaped = encoded.reshape(-1, 1)
            z_scaler = StandardScaler()
            minmax_scaler = MinMaxScaler()
            normalized = z_scaler.fit_transform(encoded_reshaped)
            scaled = minmax_scaler.fit_transform(normalized)
            
            # Store the scalers
            self.scalers[f'{cat_col}_z'] = z_scaler
            self.scalers[f'{cat_col}_minmax'] = minmax_scaler
            
            # Store encoded and normalized values
            df[f'{cat_col}_encoded'] = scaled.ravel()
            
            # Log unique categories
            n_categories = len(encoder.classes_)
            logging.info(f"Encoded {cat_col} with {n_categories} unique categories")
        
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
        
        # Create splits
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        test_df = df.iloc[test_idx]
        
        # Process features for each split
        train_features, train_labels = self._process_split(train_df, is_training=True)
        valid_features, valid_labels = self._process_split(valid_df, is_training=False)
        test_features, test_labels = self._process_split(test_df, is_training=False)
        
        # Save processed features
        output_dir = os.path.join(os.path.dirname(data_path), 'processed_features')
        self._save_processed_features(train_features, train_df, 'train', output_dir)
        self._save_processed_features(valid_features, valid_df, 'valid', output_dir)
        self._save_processed_features(test_features, test_df, 'test', output_dir)
        logging.info(f"Saved processed features to {output_dir}")
        
        # Save processed dataset with split labels
        output_path = os.path.join(os.path.dirname(data_path), 'processed_features_with_splits.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed dataset with split labels to {output_path}")
        
        # Create datasets
        datasets = {
            'train': GPCRDataset(train_features, train_labels, self.active_feature_groups, self.config, is_training=True),
            'valid': GPCRDataset(valid_features, valid_labels, self.active_feature_groups, self.config, is_training=False),
            'test': GPCRDataset(test_features, test_labels, self.active_feature_groups, self.config, is_training=False)
        }
        
        return datasets, self.scalers

    def _split_by_umap(self, 
                      df: pd.DataFrame,
                      test_size: float,
                      valid_size: float,
                      random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data using UMAP coordinates to ensure equidistant distribution in each split.
        Combines polar coordinate distribution with cluster awareness.
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Get UMAP coordinates
        umap_coords = df[['nmfUMAP1_af_qc', 'nmfUMAP2_af_qc']].values
        
        # Scale coordinates for DBSCAN
        scaler = StandardScaler()
        umap_scaled = scaler.fit_transform(umap_coords)
        
        # Identify clusters using DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        cluster_labels = dbscan.fit_predict(umap_scaled)
        n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
        n_noise = len(cluster_labels[cluster_labels == -1])
        
        logging.info(f"\nCluster Analysis:")
        logging.info(f"Found {n_clusters} clusters and {n_noise} noise points")
        
        # Calculate cluster sizes and identify small clusters
        cluster_sizes = {}
        small_clusters = set()
        for i in range(-1, max(cluster_labels) + 1):
            size = np.sum(cluster_labels == i)
            cluster_sizes[i] = size
            if size < 10 and i >= 0:  # Don't include noise points (-1)
                small_clusters.add(i)
            logging.info(f"Cluster {i}: {size} points")
        
        # Calculate target sizes for each split
        n_total = len(df)
        n_test = int(n_total * test_size)
        n_valid = int(n_total * valid_size)
        n_train = n_total - n_test - n_valid
        
        # Normalize UMAP coordinates to [0,1] range
        umap_min = umap_coords.min(axis=0)
        umap_max = umap_coords.max(axis=0)
        umap_range = umap_max - umap_min
        umap_normalized = (umap_coords - umap_min) / umap_range
        
        # Calculate polar coordinates for better spatial distribution
        center = np.mean(umap_normalized, axis=0)
        relative_coords = umap_normalized - center
        angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
        distances = np.sqrt(np.sum(relative_coords**2, axis=1))
        
        # Sort by angle and then by distance for systematic selection
        sort_idx = np.lexsort((distances, angles))
        
        # Initialize arrays for each split
        test_mask = np.zeros(n_total, dtype=bool)
        valid_mask = np.zeros(n_total, dtype=bool)
        
        # First, handle small clusters
        np.random.seed(random_state)
        for cluster_idx in small_clusters:
            cluster_mask = cluster_labels == cluster_idx
            cluster_points = np.where(cluster_mask)[0]
            
            if len(cluster_points) > 0:
                # Ensure at least one point from small clusters in each split
                n_points = len(cluster_points)
                n_per_split = max(1, n_points // 3)
                
                # Randomly assign points to splits
                np.random.shuffle(cluster_points)
                test_points = cluster_points[:n_per_split]
                valid_points = cluster_points[n_per_split:2*n_per_split]
                
                test_mask[test_points] = True
                valid_mask[valid_points] = True
                
                # Create synthetic points for small clusters
                points_needed = max(0, 3 - n_points)  # Ensure at least 3 points per split
                if points_needed > 0:
                    noise = np.random.normal(0, 0.1, (points_needed, 2))
                    base_point = umap_coords[cluster_points[0]]
                    synthetic_points = np.array([base_point + n for n in noise])
                    umap_coords = np.vstack([umap_coords, synthetic_points])
                    cluster_labels = np.append(cluster_labels, [cluster_idx] * points_needed)
        
        # Remove small cluster points from sort_idx
        remaining_mask = ~np.isin(sort_idx, np.where(np.isin(cluster_labels, list(small_clusters)))[0])
        remaining_sort_idx = sort_idx[remaining_mask]
        
        # Calculate remaining points needed for each split
        n_test_remaining = n_test - np.sum(test_mask)
        n_valid_remaining = n_valid - np.sum(valid_mask)
        total_remaining = n_test_remaining + n_valid_remaining
        
        # Distribute remaining points using polar coordinates
        if total_remaining > 0:
            step_size = len(remaining_sort_idx) / total_remaining
            for i in range(total_remaining):
                idx = remaining_sort_idx[int(i * step_size)]
                if i < n_test_remaining:
                    test_mask[idx] = True
                else:
                    valid_mask[idx] = True
        
        # Remaining points go to train
        train_mask = ~(test_mask | valid_mask)
        
        # Get indices for each split
        train_idx = df.index[train_mask]
        valid_idx = df.index[valid_mask]
        test_idx = df.index[test_mask]
        
        # Verify class balance and distribution in each split
        logging.info("\nSplit distribution:")
        for split_name, split_idx in [('Train', train_idx), ('test', test_idx), ('valid', valid_idx)]:
            n_known = (df.loc[split_idx, 'known_pair'] == 1).sum()
            n_unknown = (df.loc[split_idx, 'known_pair'] == 0).sum()
            
            # Calculate UMAP statistics for this split
            split_coords = umap_coords[np.isin(df.index, split_idx)]
            umap1_mean = np.mean(split_coords[:, 0])
            umap2_mean = np.mean(split_coords[:, 1])
            umap1_std = np.std(split_coords[:, 0])
            umap2_std = np.std(split_coords[:, 1])
            
            # Calculate cluster representation
            split_clusters = cluster_labels[np.isin(df.index, split_idx)]
            unique_clusters = len(set(split_clusters[split_clusters >= 0]))
            small_clusters_represented = len(set(split_clusters) & small_clusters)
            
            logging.info(f"\n{split_name} split:")
            logging.info(f"  Total: {len(split_idx)} samples")
            logging.info(f"  Known pairs: {n_known}")
            logging.info(f"  Unknown pairs: {n_unknown}")
            logging.info(f"  Known ratio: {n_known/len(split_idx):.2f}")
            logging.info(f"  UMAP1 mean: {umap1_mean:.2f}, std: {umap1_std:.2f}")
            logging.info(f"  UMAP2 mean: {umap2_mean:.2f}, std: {umap2_std:.2f}")
            logging.info(f"  Clusters represented: {unique_clusters}/{n_clusters}")
            logging.info(f"  Small clusters represented: {small_clusters_represented}/{len(small_clusters)}")
        
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
    
    def _clean_text(self, text):
        """Clean text by removing HTML tags and normalizing whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def _process_split(self,
                      df: pd.DataFrame,
                      is_training: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Process features for a data split."""
        features = {}
        column_groups = self._get_column_groups()
        
        # Process each feature group with appropriate normalization
        for group_name, columns in column_groups.items():
            if group_name in self.active_feature_groups:
                feature_data = df[columns].values
                
                if group_name in ['residue_contacts', 'ligand_contact_sum', 'ligand_contact_indiv']:
                    # No normalization needed, already 0-1
                    features[group_name] = feature_data
                
                elif group_name == 'expression_features':
                    # Take absolute value, then Z-score normalize and scale to 0-1
                    feature_data = np.abs(feature_data)
                    if is_training:
                        z_scaler = StandardScaler()
                        minmax_scaler = MinMaxScaler()
                        normalized = z_scaler.fit_transform(feature_data)
                        features[group_name] = minmax_scaler.fit_transform(normalized)
                        self.scalers[f'{group_name}_z'] = z_scaler
                        self.scalers[f'{group_name}_minmax'] = minmax_scaler
                    else:
                        normalized = self.scalers[f'{group_name}_z'].transform(feature_data)
                        features[group_name] = self.scalers[f'{group_name}_minmax'].transform(normalized)
                
                else:
                    # Z-score normalize and scale to 0-1 for all other features
                    if is_training:
                        z_scaler = StandardScaler()
                        minmax_scaler = MinMaxScaler()
                        normalized = z_scaler.fit_transform(feature_data)
                        features[group_name] = minmax_scaler.fit_transform(normalized)
                        self.scalers[f'{group_name}_z'] = z_scaler
                        self.scalers[f'{group_name}_minmax'] = minmax_scaler
                    else:
                        normalized = self.scalers[f'{group_name}_z'].transform(feature_data)
                        features[group_name] = self.scalers[f'{group_name}_minmax'].transform(normalized)
        
        # Get categorical features that were already encoded
        for cat_col in self.config['categorical_columns']:
            features[f'{cat_col}_encoded'] = df[f'{cat_col}_encoded'].values
        
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
                # Z-score normalize and scale to 0-1
                receptor_z_scaler = StandardScaler()
                receptor_minmax_scaler = MinMaxScaler()
                ligand_z_scaler = StandardScaler()
                ligand_minmax_scaler = MinMaxScaler()
                
                # Normalize receptor profiles
                normalized_receptor = receptor_z_scaler.fit_transform(receptor_profiles)
                features['receptor_expression'] = receptor_minmax_scaler.fit_transform(normalized_receptor)
                
                # Normalize ligand profiles
                normalized_ligand = ligand_z_scaler.fit_transform(ligand_profiles)
                features['ligand_expression'] = ligand_minmax_scaler.fit_transform(normalized_ligand)
                
                # Store scalers
                self.scalers['receptor_expression_z'] = receptor_z_scaler
                self.scalers['receptor_expression_minmax'] = receptor_minmax_scaler
                self.scalers['ligand_expression_z'] = ligand_z_scaler
                self.scalers['ligand_expression_minmax'] = ligand_minmax_scaler
            else:
                # Transform using fitted scalers
                normalized_receptor = self.scalers['receptor_expression_z'].transform(receptor_profiles)
                features['receptor_expression'] = self.scalers['receptor_expression_minmax'].transform(normalized_receptor)
                
                normalized_ligand = self.scalers['ligand_expression_z'].transform(ligand_profiles)
                features['ligand_expression'] = self.scalers['ligand_expression_minmax'].transform(normalized_ligand)
        
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