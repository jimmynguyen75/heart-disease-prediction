"""
Data Preprocessing Module
Handles data loading, cleaning, feature engineering, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, random_state=123):
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, file_path):
        """Load CSV data"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_data_summary(self, df):
        """Get comprehensive data summary"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'statistics': df.describe().to_dict()
        }
        return summary
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values based on strategy"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64']:
                    if strategy == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif strategy == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def detect_target_column(self, df):
        """Auto-detect target column"""
        target_keywords = ['target', 'Target', 'label', 'Label', 'class', 'Class']
        for col in df.columns:
            if col in target_keywords:
                return col
        return df.columns[-1]
    
    def get_correlation_matrix(self, df):
        """Calculate and return correlation matrix"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        return corr_matrix
    
    def plot_correlation_heatmap(self, df):
        """Generate correlation heatmap"""
        corr_matrix = self.get_correlation_matrix(df)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def split_data(self, df, target_col, test_size=0.2, scale_method='standard', feature_list=None):
        """
        Split data into train/test sets following paper's methodology

        Paper's approach (Reference paper):
        - 80/20 split with set.seed(123) in R
        - Stratified sampling via caret package
        - Scaling for distance-based models (KNN, SVM)

        Our implementation:
        - 80/20 split with random_state=123 (equivalent to set.seed)
        - Stratified sampling via sklearn's stratify parameter
        - StandardScaler for numerical features (matching paper's scale())
        """
        if feature_list is not None:
            X = df[feature_list]
        else:
            # 6 features selected as per reference paper
            paper_features = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
            available_features = [f for f in paper_features if f in df.columns]
            X = df[available_features] if available_features else df.drop(columns=[target_col])
        y = df[target_col]

        self.feature_names = X.columns.tolist()
        self.target_name = target_col

        # Stratified split (matches paper's caret stratification)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y, shuffle=True  # Explicitly shuffle before split
        )

        # Scale if needed (matches paper's scale() for KNN/SVM)
        if scale_method == 'standard':
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler()
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        # else: no scaling (for tree-based models)

        return X_train, X_test, y_train, y_test


def prepare_features_for_aco(df, target_col):
    """Return X (all features) and y for ACO to search over."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
