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
import matplotlib.pyplot as plt

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
                 data_config: Dict,
                 codes: Optional[np.ndarray] = None,
                 is_training: bool = True):
        """Initialize dataset.
        
        Args:
            features: Dictionary of feature arrays
            labels: Array of labels
            active_feature_groups: Set of active feature groups
            data_config: Data configuration dictionary
            codes: Optional array of code identifiers
            is_training: Whether this is training data
        """
        self.features = features
        self.labels = labels
        self.active_feature_groups = active_feature_groups
        self.data_config = data_config
        self.codes = codes
        self.is_training = is_training
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get features based on active feature groups
        feature_dict = {}
        
        # Get standard feature groups
        for group in self.active_feature_groups:
            if group in self.features:
                feature_data = self.features[group][idx]
                if isinstance(feature_data, (list, np.ndarray)):
                    feature_data = np.array(feature_data)
                # Ensure 2D array for tensor conversion
                if feature_data.ndim == 1:
                    feature_data = feature_data.reshape(1, -1)
                feature_dict[group] = torch.FloatTensor(feature_data)
        
        # Get individual categorical features
        for cat_col in self.data_config['categorical_columns']:
            if f'{cat_col}_encoded' in self.features:
                feature_data = self.features[f'{cat_col}_encoded'][idx]
                # Convert single float to array
                if isinstance(feature_data, (float, np.float64)):
                    feature_data = np.array([feature_data])
                elif isinstance(feature_data, (list, np.ndarray)):
                    feature_data = np.array(feature_data)
                # Ensure 2D array for tensor conversion
                if feature_data.ndim == 1:
                    feature_data = feature_data.reshape(1, -1)
                feature_dict[f'{cat_col}_encoded'] = torch.FloatTensor(feature_data)
        
        # Get label
        label = torch.FloatTensor([self.labels[idx]])
        
        # Get code if available
        if self.codes is not None:
            feature_dict['code'] = self.codes[idx]
        
        return feature_dict, label

class DataPreprocessor:
    """Handles data preprocessing for GPCR-Ligand binding prediction."""
    
    def __init__(self, data_config_path: str, model_config_path: str):
        """
        Args:
            data_config_path: Path to data configuration file containing column mappings
            model_config_path: Path to model configuration file containing model and training parameters
        """
        self.data_config = self._load_data_config(data_config_path)
        self.model_config = self._load_model_config(model_config_path)
        self.scalers = {}
        self.encoders = {}
        self.expression_cache = {}
        self.active_feature_groups = {
            group for group, active in self.model_config['feature_groups'].items() 
            if active
        }
        logging.info("Active feature groups:")
        for group in sorted(self.active_feature_groups):
            logging.info(f"  - {group}")
        
    def _load_data_config(self, config_path: str) -> dict:
        """Load data configuration file containing column mappings."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _load_model_config(self, config_path: str) -> dict:
        """Load model configuration file containing model and training parameters."""
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
                if config_key in self.data_config:
                    column_groups[group] = self.data_config[config_key]
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
    
    def balance_and_split_dataset(self, df: pd.DataFrame, split_method: str = 'cluster', test_size: float = 0.2, valid_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Balance the dataset and create train/valid/test splits in a single operation.
        Ensures all models with the same afpd_dir_name stay in the same split.
        
        Args:
            df: Input DataFrame
            split_method: Method for splitting ('cluster' or 'random_balanced')
            test_size: Proportion of data for test set
            valid_size: Proportion of data for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        np.random.seed(random_state)
        
        # Validate required columns
        if 'afpd_dir_name' not in df.columns:
            raise ValueError("Column 'afpd_dir_name' not found in dataframe")
        
        # Analyze group structure
        group_sizes = df.groupby('afpd_dir_name').size()
        logging.info(f"\nGroup structure analysis:")
        logging.info(f"  Total groups: {len(group_sizes)}")
        logging.info(f"  Models per group distribution: {group_sizes.value_counts().sort_index().to_dict()}")
        logging.info(f"  Average models per group: {group_sizes.mean():.1f}")
        
        # Separate groups by known/unknown status
        group_labels = {}
        mixed_groups = []
        
        for group_name in df['afpd_dir_name'].unique():
            group_data = df[df['afpd_dir_name'] == group_name]
            unique_labels = group_data['known_pair'].unique()
            
            if len(unique_labels) == 1:
                group_labels[group_name] = unique_labels[0]
            else:
                mixed_groups.append(group_name)
                group_labels[group_name] = group_data['known_pair'].iloc[0]
        
        if mixed_groups:
            logging.warning(f"Found {len(mixed_groups)} groups with mixed known/unknown labels")
        
        # Get known and unknown groups
        known_groups = [group for group, label in group_labels.items() if label == 1]
        unknown_groups = [group for group, label in group_labels.items() if label == 0]
        
        # Get ALL models from known groups
        all_known_indices = df[df['afpd_dir_name'].isin(known_groups)].index
        n_known_models = len(all_known_indices)
        n_known_groups = len(known_groups)
        
        if n_known_groups == 0:
            logging.warning("No known groups found!")
            return df, df, df
            
        # Calculate target sizes
        unknown_groups_needed = n_known_groups  # Same number of groups for 1:1 model ratio
        
        logging.info(f"\nBalance analysis:")
        logging.info(f"  Known groups: {n_known_groups}")
        logging.info(f"  Known models: {n_known_models}")
        logging.info(f"  Available unknown groups: {len(unknown_groups)}")
        logging.info(f"  Unknown groups needed: {unknown_groups_needed}")
        
        if len(unknown_groups) == 0:
            logging.warning("No unknown groups found, returning only known samples")
            return df.loc[all_known_indices], df.loc[all_known_indices], df.loc[all_known_indices]
        
        if len(unknown_groups) < unknown_groups_needed:
            logging.warning(f"Not enough unknown groups ({len(unknown_groups)}) to match known groups ({unknown_groups_needed})")
            unknown_groups_needed = len(unknown_groups)
        
        if split_method == 'cluster':
            # Get cluster column from config
            cluster_column = self.model_config.get('cluster_column', 'cluster_id')
            if cluster_column not in df.columns:
                raise ValueError(f"Cluster column '{cluster_column}' not found in dataframe")
            
            # Get cluster information for unknown groups
            unknown_df = df[df['afpd_dir_name'].isin(unknown_groups)].copy()
            unknown_df['group_name'] = unknown_df['afpd_dir_name']
            
            # Get unique clusters
            unique_clusters = unknown_df[cluster_column].unique()
            logging.info(f"\nFound {len(unique_clusters)} unique clusters in unknown groups")
            
            # Calculate target groups per cluster
            target_per_cluster = max(1, unknown_groups_needed // len(unique_clusters))
            remaining_needed = unknown_groups_needed
            
            # Select groups from each cluster
            selected_groups = []
            cluster_selections = {}
            
            for cluster_id in unique_clusters:
                cluster_groups = unknown_df[unknown_df[cluster_column] == cluster_id]
                if len(cluster_groups) > 0:
                    # Calculate how many to select from this cluster
                    n_to_select = min(target_per_cluster, len(cluster_groups), remaining_needed)
                    if n_to_select > 0:
                        selected = cluster_groups.sample(n=n_to_select, random_state=random_state)
                        selected_groups.extend(selected['group_name'].unique().tolist())
                        remaining_needed -= n_to_select
                        cluster_selections[cluster_id] = n_to_select
            
            # If we still need more groups, randomly select from remaining groups
            if len(selected_groups) < unknown_groups_needed:
                remaining_groups = [g for g in unknown_groups if g not in selected_groups]
                n_to_select = unknown_groups_needed - len(selected_groups)
                additional_selected = np.random.choice(remaining_groups, size=n_to_select, replace=False)
                selected_groups.extend(additional_selected)
            
            selected_unknown_groups = selected_groups[:unknown_groups_needed]
            
            # Log selection results
            logging.info(f"\nSelected unknown groups:")
            logging.info(f"  Total selected: {len(selected_unknown_groups)}")
            logging.info("\nCluster selection details:")
            for cluster_id, count in cluster_selections.items():
                original_size = len(unknown_df[unknown_df[cluster_column] == cluster_id])
                selection_ratio = count / original_size
                logging.info(f"  Cluster {cluster_id}: Selected {count}/{original_size} points ({selection_ratio:.1%})")
            
            # Create selected groups list
            selected_groups = known_groups + list(selected_unknown_groups)
            
            # Create a dataframe with just the selected groups
            selected_df = df[df['afpd_dir_name'].isin(selected_groups)].copy()
            
            # Split at the GROUP level to ensure group integrity
            # Get unique groups and their labels
            group_info = selected_df.groupby('afpd_dir_name')['known_pair'].first().reset_index()
            known_group_names = group_info[group_info['known_pair'] == 1]['afpd_dir_name'].tolist()
            unknown_group_names = group_info[group_info['known_pair'] == 0]['afpd_dir_name'].tolist()
            
            # Calculate split sizes for groups (not individual models)
            n_known_groups_total = len(known_group_names)
            n_unknown_groups_total = len(unknown_group_names)
            
            n_known_test = int(n_known_groups_total * test_size)
            n_known_valid = int(n_known_groups_total * valid_size)
            n_known_train = n_known_groups_total - n_known_test - n_known_valid
            
            n_unknown_test = int(n_unknown_groups_total * test_size)
            n_unknown_valid = int(n_unknown_groups_total * valid_size)
            n_unknown_train = n_unknown_groups_total - n_unknown_test - n_unknown_valid
            
            # Split known groups
            known_groups_shuffled = np.random.permutation(known_group_names)
            known_train_groups = known_groups_shuffled[:n_known_train]
            known_valid_groups = known_groups_shuffled[n_known_train:n_known_train + n_known_valid]
            known_test_groups = known_groups_shuffled[n_known_train + n_known_valid:]
            
            # Split unknown groups
            unknown_groups_shuffled = np.random.permutation(unknown_group_names)
            unknown_train_groups = unknown_groups_shuffled[:n_unknown_train]
            unknown_valid_groups = unknown_groups_shuffled[n_unknown_train:n_unknown_train + n_unknown_valid]
            unknown_test_groups = unknown_groups_shuffled[n_unknown_train + n_unknown_valid:]
            
            # Combine group splits
            train_groups = list(known_train_groups) + list(unknown_train_groups)
            valid_groups = list(known_valid_groups) + list(unknown_valid_groups)
            test_groups = list(known_test_groups) + list(unknown_test_groups)
            
            # Create split dataframes by filtering on group names
            train_df = df[df['afpd_dir_name'].isin(train_groups)].copy()
            valid_df = df[df['afpd_dir_name'].isin(valid_groups)].copy()
            test_df = df[df['afpd_dir_name'].isin(test_groups)].copy()
            
            # Verify balance in each split
            for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
                n_known = (split_df['known_pair'] == 1).sum()
                n_unknown = (split_df['known_pair'] == 0).sum()
                total = len(split_df)
                known_ratio = n_known / total
                n_groups = split_df['afpd_dir_name'].nunique()
                logging.info(f"\n{split_name} split balance:")
                logging.info(f"  Known pairs: {n_known} ({known_ratio:.2%})")
                logging.info(f"  Unknown pairs: {n_unknown} ({1-known_ratio:.2%})")
                logging.info(f"  Total samples: {total}")
                logging.info(f"  Number of groups: {n_groups}")
            
            return train_df, valid_df, test_df
            
        elif split_method == 'random_balanced':
            # Select unknown groups randomly
            selected_unknown_groups = np.random.choice(unknown_groups, size=unknown_groups_needed, replace=False)
            
            # Get all indices for selected groups
            selected_indices = df[df['afpd_dir_name'].isin(known_groups + list(selected_unknown_groups))].index
            
            # Create a dataframe with just the selected groups
            selected_df = df.loc[selected_indices].copy()
            
            # Split at the GROUP level to ensure group integrity
            # Get unique groups and their labels
            group_info = selected_df.groupby('afpd_dir_name')['known_pair'].first().reset_index()
            known_group_names = group_info[group_info['known_pair'] == 1]['afpd_dir_name'].tolist()
            unknown_group_names = group_info[group_info['known_pair'] == 0]['afpd_dir_name'].tolist()
            
            # Calculate split sizes for groups (not individual models)
            n_known_groups_total = len(known_group_names)
            n_unknown_groups_total = len(unknown_group_names)
            
            n_known_test = int(n_known_groups_total * test_size)
            n_known_valid = int(n_known_groups_total * valid_size)
            n_known_train = n_known_groups_total - n_known_test - n_known_valid
            
            n_unknown_test = int(n_unknown_groups_total * test_size)
            n_unknown_valid = int(n_unknown_groups_total * valid_size)
            n_unknown_train = n_unknown_groups_total - n_unknown_test - n_unknown_valid
            
            # Split known groups
            known_groups_shuffled = np.random.permutation(known_group_names)
            known_train_groups = known_groups_shuffled[:n_known_train]
            known_valid_groups = known_groups_shuffled[n_known_train:n_known_train + n_known_valid]
            known_test_groups = known_groups_shuffled[n_known_train + n_known_valid:]
            
            # Split unknown groups
            unknown_groups_shuffled = np.random.permutation(unknown_group_names)
            unknown_train_groups = unknown_groups_shuffled[:n_unknown_train]
            unknown_valid_groups = unknown_groups_shuffled[n_unknown_train:n_unknown_train + n_unknown_valid]
            unknown_test_groups = unknown_groups_shuffled[n_unknown_train + n_unknown_valid:]
            
            # Combine group splits
            train_groups = list(known_train_groups) + list(unknown_train_groups)
            valid_groups = list(known_valid_groups) + list(unknown_valid_groups)
            test_groups = list(known_test_groups) + list(unknown_test_groups)
            
            # Create split dataframes by filtering on group names
            train_df = df[df['afpd_dir_name'].isin(train_groups)].copy()
            valid_df = df[df['afpd_dir_name'].isin(valid_groups)].copy()
            test_df = df[df['afpd_dir_name'].isin(test_groups)].copy()
            
            # Verify balance in each split
            for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
                n_known = (split_df['known_pair'] == 1).sum()
                n_unknown = (split_df['known_pair'] == 0).sum()
                total = len(split_df)
                known_ratio = n_known / total
                n_groups = split_df['afpd_dir_name'].nunique()
                logging.info(f"\n{split_name} split balance:")
                logging.info(f"  Known pairs: {n_known} ({known_ratio:.2%})")
                logging.info(f"  Unknown pairs: {n_unknown} ({1-known_ratio:.2%})")
                logging.info(f"  Total samples: {total}")
                logging.info(f"  Number of groups: {n_groups}")
            
            return train_df, valid_df, test_df

    def _save_processed_features(self, features: Dict[str, np.ndarray], df: pd.DataFrame, split: str, output_dir: str):
        """Save processed features to CSV files.
        
        Args:
            features: Dictionary of processed features
            df: Original dataframe
            split: Split name (train/valid/test)
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each feature group separately (normalized values only)
        for group_name, feature_array in features.items():
            if group_name.endswith('_encoded'):
                # For encoded categorical features, save both encoded and original values
                cat_col = group_name.replace('_encoded', '')
                encoded_df = pd.DataFrame({
                    'code': df['code'].values,  # Use .values to ensure we get the array
                    'original_value': df[cat_col].values,
                    'encoded_value': feature_array,
                    'cleaned_value': df[cat_col].apply(self._clean_text).values
                })
                encoded_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}.csv'), index=False)
                logging.info(f"Saved {split} {group_name} to {output_dir}")
            else:
                # For normalized features, get original column names from config
                if group_name in self._get_column_groups():
                    columns = self._get_column_groups()[group_name]
                    feature_df = pd.DataFrame(feature_array, columns=columns)
                    # Add code column as first column using .values
                    feature_df.insert(0, 'code', df['code'].values)
                    feature_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}_normalized.csv'), index=False)
                    logging.info(f"Saved {split} {group_name} normalized features to {output_dir}")
                else:
                    # For other features (like expression profiles)
                    feature_df = pd.DataFrame(feature_array)
                    # Add code column as first column using .values
                    feature_df.insert(0, 'code', df['code'].values)
                    feature_df.to_csv(os.path.join(output_dir, f'{split}_{group_name}.csv'), index=False)
                    logging.info(f"Saved {split} {group_name} to {output_dir}")
            
            # Verify code column was copied correctly and print first 5 codes
            saved_file = os.path.join(output_dir, f'{split}_{group_name}{"_normalized" if group_name in self._get_column_groups() else ""}.csv')
            if os.path.exists(saved_file):
                saved_df = pd.read_csv(saved_file)
                if 'code' not in saved_df.columns:
                    logging.error(f"Code column missing in saved file: {saved_file}")
                elif saved_df['code'].isna().any():
                    logging.error(f"Found null values in code column of saved file: {saved_file}")
                elif (saved_df['code'] == '').any():
                    logging.error(f"Found empty strings in code column of saved file: {saved_file}")
                else:
                    logging.info(f"Verified code column in {saved_file}: {len(saved_df)} rows")
                    # Print first 5 codes
                    first_5_codes = saved_df['code'].head().tolist()
                    logging.info(f"First 5 codes in {saved_file}:")
                    for i, code in enumerate(first_5_codes, 1):
                        logging.info(f"  {i}. {code}")

    def preprocess_data(self, 
                       data_path: Optional[str] = None,
                       split_method: str = 'cluster',
                       test_size: Optional[float] = None,
                       valid_size: Optional[float] = None,
                       random_state: int = 42,
                       feature_groups: Optional[Dict[str, bool]] = None,
                       filters: Optional[Dict[str, Union[str, List[str], Dict[str, bool]]]] = None) -> Tuple[Dict[str, Dataset], Dict[str, StandardScaler]]:
        """
        Preprocess data and create train/valid/test splits.
        """
        # Use config values if not provided
        data_path = data_path or self.model_config['data_path']
        test_size = test_size or self.model_config['data_params']['test_size']
        valid_size = valid_size or self.model_config['data_params']['valid_size']
        random_state = random_state or self.model_config['data_params'].get('random_state', 42)
        filters = filters or self.model_config.get('filters')
        
        # Get split method from config if not provided
        if split_method == 'cluster' and 'split_method' in self.model_config['data_params']:
            split_method = self.model_config['data_params']['split_method']
            logging.info(f"Using split method from config: {split_method}")
        
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
        df = pd.read_csv(data_path, low_memory=False)  # Add low_memory=False to handle mixed types
        
        # Check for code column
        if 'code' not in df.columns:
            logging.warning("No 'code' column found in the data. Creating one using afpd_dir_name...")
            df['code'] = df['afpd_dir_name']
        else:
            # Check for empty or null values in code column
            null_codes = df['code'].isna().sum()
            if null_codes > 0:
                logging.warning(f"Found {null_codes} null values in code column. Filling with afpd_dir_name...")
                df.loc[df['code'].isna(), 'code'] = df.loc[df['code'].isna(), 'afpd_dir_name']
            
            # Check for empty strings
            empty_codes = (df['code'] == '').sum()
            if empty_codes > 0:
                logging.warning(f"Found {empty_codes} empty strings in code column. Filling with afpd_dir_name...")
                df.loc[df['code'] == '', 'code'] = df.loc[df['code'] == '', 'afpd_dir_name']
        
        # Log code column statistics
        unique_codes = df['code'].nunique()
        logging.info(f"Code column statistics:")
        logging.info(f"  Total rows: {len(df)}")
        logging.info(f"  Unique codes: {unique_codes}")
        logging.info(f"  Null values: {df['code'].isna().sum()}")
        logging.info(f"  Empty strings: {(df['code'] == '').sum()}")
        
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
            
            # Check code column after filtering
            logging.info(f"Code column statistics after filtering:")
            logging.info(f"  Total rows: {len(df)}")
            logging.info(f"  Unique codes: {df['code'].nunique()}")
            logging.info(f"  Null values: {df['code'].isna().sum()}")
            logging.info(f"  Empty strings: {(df['code'] == '').sum()}")
        
        # Pre-process categorical features before splitting
        for cat_col in self.data_config['categorical_columns']:
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
        
        # Balance and split dataset in one operation
        train_df, valid_df, test_df = self.balance_and_split_dataset(
            df,
            split_method=split_method,
            test_size=test_size,
            valid_size=valid_size,
            random_state=random_state
        )
        
        # Add split labels to each dataframe
        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        
        # Combine all splits back into one dataframe for saving
        df_with_splits = pd.concat([train_df, valid_df, test_df])
        
        # Process features for each split
        train_features, train_labels, train_codes = self._process_split(train_df, is_training=True)
        valid_features, valid_labels, valid_codes = self._process_split(valid_df, is_training=False)
        test_features, test_labels, test_codes = self._process_split(test_df, is_training=False)
        
        # Save processed features using config name
        config_name = self.model_config.get('name', 'unnamed_config')
        output_dir = os.path.join('data', 'preprocessing_tests', config_name)
        self._save_processed_features(train_features, train_df, 'train', output_dir)
        self._save_processed_features(valid_features, valid_df, 'valid', output_dir)
        self._save_processed_features(test_features, test_df, 'test', output_dir)
        df_with_splits.to_csv(os.path.join(output_dir, 'df_with_splits.csv'), index=False)
        logging.info(f"Saved processed features to {output_dir}")
        
        # Create joined file with original data and splits
        original_data_path = self.model_config['data_path']
        original_df = pd.read_csv(original_data_path, low_memory=False)  # Add low_memory=False to handle mixed types
        
        # Create a dataframe with just the split information
        splits_df = pd.DataFrame({
            'afpd_dir_name': df_with_splits['afpd_dir_name'],
            'split': df_with_splits['split']
        })
        
        # Left join original data with splits
        joined_df = original_df.merge(splits_df, on='afpd_dir_name', how='left')  # Changed from 'left' to 'inner' join
        
        # Save joined file
        original_filename = os.path.basename(original_data_path)
        base_name = os.path.splitext(original_filename)[0]
        joined_path = os.path.join(output_dir, f'{base_name}_with_splits.csv')
        joined_df.to_csv(joined_path, index=False)
        logging.info(f"Saved joined file with splits to: {joined_path}")
        logging.info(f"Total rows in joined file: {len(joined_df)}")
        
        # Create datasets
        datasets = {
            'train': GPCRDataset(train_features, train_labels, self.active_feature_groups, self.data_config, codes=train_codes, is_training=True),
            'valid': GPCRDataset(valid_features, valid_labels, self.active_feature_groups, self.data_config, codes=valid_codes, is_training=False),
            'test': GPCRDataset(test_features, test_labels, self.active_feature_groups, self.data_config, codes=test_codes, is_training=False)
        }
        
        return datasets, self.scalers

    def _clean_text(self, text):
        """Clean text by removing HTML tags and normalizing whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def _process_split(self,
                      df: pd.DataFrame,
                      is_training: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]:
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
        for cat_col in self.data_config['categorical_columns']:
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
        
        # Get codes if available
        codes = df['code'].values if 'code' in df.columns else None
        
        return features, labels, codes

    def preprocess_data_multiple_rounds(self, 
                                      data_path: Optional[str] = None,
                                      split_method: str = 'umap',
                                      test_size: Optional[float] = None,
                                      valid_size: Optional[float] = None,
                                      random_state: int = 42,
                                      feature_groups: Optional[Dict[str, bool]] = None,
                                      filters: Optional[Dict[str, Union[str, List[str], Dict[str, bool]]]] = None,
                                      n_rounds: int = 5) -> List[Tuple[Dict[str, Dataset], Dict[str, StandardScaler]]]:
        """
        Preprocess data and create multiple training rounds with different unknown pairs.
        
        Args:
            data_path: Path to data file
            split_method: Method for splitting ('umap', 'random', or 'random_balanced')
            test_size: Proportion of data for test set
            valid_size: Proportion of data for validation set
            random_state: Random seed
            feature_groups: Dictionary of feature groups to use
            filters: Dictionary of filters to apply
            n_rounds: Number of training rounds to perform
            
        Returns:
            List of tuples containing (datasets, scalers) for each round
        """
        # Use config values if not provided
        data_path = data_path or self.model_config['data_path']
        test_size = test_size or self.model_config['data_params']['test_size']
        valid_size = valid_size or self.model_config['data_params']['valid_size']
        random_state = random_state or self.model_config['data_params'].get('random_state', 42)
        filters = filters or self.model_config.get('filters')
        
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
        df = pd.read_csv(data_path, low_memory=False)
        
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
        
        # Pre-process categorical features
        for cat_col in self.data_config['categorical_columns']:
            df[f'{cat_col}_cleaned'] = df[cat_col].apply(self._clean_text)
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(df[f'{cat_col}_cleaned'])
            self.encoders[cat_col] = encoder
            encoded_reshaped = encoded.reshape(-1, 1)
            z_scaler = StandardScaler()
            minmax_scaler = MinMaxScaler()
            normalized = z_scaler.fit_transform(encoded_reshaped)
            scaled = minmax_scaler.fit_transform(normalized)
            self.scalers[f'{cat_col}_z'] = z_scaler
            self.scalers[f'{cat_col}_minmax'] = minmax_scaler
            df[f'{cat_col}_encoded'] = scaled.ravel()
            n_categories = len(encoder.classes_)
            logging.info(f"Encoded {cat_col} with {n_categories} unique categories")
        
        # Separate known and unknown pairs
        known_df = df[df['known_pair'] == 1].copy()
        unknown_df = df[df['known_pair'] == 0].copy()
        
        # Get unique groups for known and unknown pairs
        known_groups = known_df['afpd_dir_name'].unique()
        unknown_groups = unknown_df['afpd_dir_name'].unique()
        
        logging.info(f"\nStarting multiple training rounds:")
        logging.info(f"  Known groups: {len(known_groups)}")
        logging.info(f"  Unknown groups: {len(unknown_groups)}")
        logging.info(f"  Number of rounds: {n_rounds}")
        
        # Calculate number of unknown groups per round
        unknown_groups_per_round = len(unknown_groups) // n_rounds
        if unknown_groups_per_round == 0:
            unknown_groups_per_round = 1
            n_rounds = len(unknown_groups)
            logging.warning(f"Not enough unknown groups for {n_rounds} rounds. Adjusting to {n_rounds} rounds.")
        
        # Store results for each round
        round_results = []
        
        # Perform multiple rounds
        for round_idx in range(n_rounds):
            logging.info(f"\nProcessing round {round_idx + 1}/{n_rounds}")
            
            # Select unknown groups for this round
            start_idx = round_idx * unknown_groups_per_round
            end_idx = start_idx + unknown_groups_per_round
            if round_idx == n_rounds - 1:  # Last round gets remaining groups
                end_idx = len(unknown_groups)
            
            selected_unknown_groups = unknown_groups[start_idx:end_idx]
            logging.info(f"  Selected {len(selected_unknown_groups)} unknown groups for this round")
            
            # Create round dataframe
            round_df = pd.concat([
                known_df,
                unknown_df[unknown_df['afpd_dir_name'].isin(selected_unknown_groups)]
            ])
            
            # Balance and split dataset
            train_df, valid_df, test_df = self.balance_and_split_dataset(
                round_df,
                split_method=split_method,
                test_size=test_size,
                valid_size=valid_size,
                random_state=random_state + round_idx  # Different seed for each round
            )
            
            # Add split labels
            train_df['split'] = 'train'
            valid_df['split'] = 'valid'
            test_df['split'] = 'test'
            
            # Combine splits for saving
            df_with_splits = pd.concat([train_df, valid_df, test_df])
            
            # Process features for each split
            train_features, train_labels, train_codes = self._process_split(train_df, is_training=True)
            valid_features, valid_labels, valid_codes = self._process_split(valid_df, is_training=False)
            test_features, test_labels, test_codes = self._process_split(test_df, is_training=False)
            
            # Save processed features
            config_name = self.model_config.get('name', 'unnamed_config')
            output_dir = os.path.join('data', 'preprocessing_tests', f"{config_name}_round_{round_idx + 1}")
            self._save_processed_features(train_features, train_df, 'train', output_dir)
            self._save_processed_features(valid_features, valid_df, 'valid', output_dir)
            self._save_processed_features(test_features, test_df, 'test', output_dir)
            df_with_splits.to_csv(os.path.join(output_dir, 'df_with_splits.csv'), index=False)
            logging.info(f"Saved processed features to {output_dir}")
            
            # Create datasets
            datasets = {
                'train': GPCRDataset(train_features, train_labels, self.active_feature_groups, self.data_config, codes=train_codes, is_training=True),
                'valid': GPCRDataset(valid_features, valid_labels, self.active_feature_groups, self.data_config, codes=valid_codes, is_training=False),
                'test': GPCRDataset(test_features, test_labels, self.active_feature_groups, self.data_config, codes=test_codes, is_training=False)
            }
            
            # Store results for this round
            round_results.append((datasets, self.scalers.copy()))
            
            # Log round statistics
            for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
                n_known = (split_df['known_pair'] == 1).sum()
                n_unknown = (split_df['known_pair'] == 0).sum()
                total = len(split_df)
                known_ratio = n_known / total
                logging.info(f"\n{split_name} split balance for round {round_idx + 1}:")
                logging.info(f"  Known pairs: {n_known} ({known_ratio:.2%})")
                logging.info(f"  Unknown pairs: {n_unknown} ({1-known_ratio:.2%})")
                logging.info(f"  Total samples: {total}")
        
        return round_results

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