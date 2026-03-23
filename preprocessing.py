"""
Data Preprocessing Module
Handles data loading, cleaning, feature engineering, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
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

        if strategy == 'knn':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Drop all-NaN numeric columns first (KNNImputer cannot handle them)
            all_nan_cols = [c for c in numeric_cols if df_clean[c].isna().all()]
            if all_nan_cols:
                df_clean = df_clean.drop(columns=all_nan_cols)
                numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            # Fill non-numeric columns with mode
            for col in df_clean.select_dtypes(exclude=[np.number]).columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_vals = df_clean[col].mode()
                    if len(mode_vals) > 0:
                        df_clean[col].fillna(mode_vals[0], inplace=True)
            return df_clean

        cols_to_drop = []
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    col_mean = df_clean[col].mean()
                    if pd.isna(col_mean):
                        # All-NaN column — drop it rather than leave NaN for scaler
                        cols_to_drop.append(col)
                    elif strategy == 'mean':
                        df_clean[col].fillna(col_mean, inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif strategy == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                else:
                    mode_vals = df_clean[col].mode()
                    if len(mode_vals) > 0:
                        df_clean[col].fillna(mode_vals[0], inplace=True)
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

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
        plt.title('Ma trận tương quan (Correlation Matrix)', fontsize=16, fontweight='bold')
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


import re as _re

def _parse_ketqua(value):
    """
    8-step conversion for Vietnamese medical ketqua (test result) values.
    Returns float or NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()

    # Step 1: Direct numeric parse
    try:
        return float(s)
    except ValueError:
        pass

    # Step 2: Negative number with extra space, e.g. "- 3.8" → -3.8
    m = _re.match(r'^-\s+([\d.]+)$', s)
    if m:
        try:
            return float('-' + m.group(1))
        except ValueError:
            pass

    # Step 3: Number with unit suffix, e.g. "225h" → 225.0
    m = _re.match(r'^([\d.]+)[a-zA-Z]+$', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Step 4: Qualitative negative → 0.0
    if s.lower() in ('negative', 'âm tính', 'am tinh', 'âm', 'am'):
        return 0.0

    # Step 5: Qualitative positive → 1.0
    if s.lower() in ('positive', 'dương tính', 'duong tinh', 'dương', 'duong'):
        return 1.0

    # Step 6: Trace → 0.5
    if s.lower() == 'trace':
        return 0.5

    # Step 7: Ordinal semi-quantitative, e.g. "1+" → 1.0, "2+" → 2.0
    m = _re.match(r'^(\d+)\+$', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Step 8: Everything else → NaN
    return np.nan


def preprocess_vn_data(df_raw):
    """
    Preprocess long-format Vietnamese medical visit data (_vn files) into
    a visit-level tabular dataset ready for ML.

    Supports two formats:
      - Format A (has 'icd_level_0', 'tenxn'): target = icd_level_0 == 'IX'
      - Format B (has 'is_direct_cardio'): target = is_direct_cardio (already binary)

    Steps:
      1. Remove duplicates, drop rows missing 'mavaovien' or non-numeric 'ketqua'
      2. Build binary target per mavaovien
      3. Pivot 'maxn' → columns, aggregate by mean per mavaovien
      4. Merge vitals & demographics (tuoi, phai + any numeric vital columns)
    """
    df = df_raw.copy()

    # --- Detect format ---
    # Format A: has 'icd_level_0' → derive target from IX
    # Format B: has 'is_direct_cardio' → use directly
    # Format C: has 'target' column already
    has_direct_target = "is_direct_cardio" in df.columns or "target" in df.columns
    direct_target_col = "target" if "target" in df.columns else "is_direct_cardio"
    has_tenxn = "tenxn" in df.columns

    # --- Basic cleaning ---
    df = df.drop_duplicates()
    df = df.dropna(subset=["mavaovien"])

    # Drop rows where BOTH maxn and ketqua are missing
    both_missing = df["maxn"].isna() & df["ketqua"].isna()
    df = df[~both_missing]

    # Convert ketqua using 8-step parsing
    df["ketqua"] = df["ketqua"].apply(_parse_ketqua)
    df = df.dropna(subset=["ketqua"])

    # --- Build binary target per mavaovien ---
    if has_direct_target:
        # Format B/C: target column already binary 0/1, take first value per visit
        visit_target = (
            df.groupby("mavaovien")[direct_target_col]
            .first()
            .rename("target")
            .reset_index()
        )
    else:
        # Format A: derive from icd_level_0 == 'IX'
        df["icd_level_0"] = df["icd_level_0"].fillna("unknown")
        visit_target = (
            df.groupby("mavaovien")["icd_level_0"]
            .apply(lambda x: int((x == "IX").any()))
            .rename("target")
            .reset_index()
        )

    # --- Pivot maxn → columns (mean per visit) ---
    pivot = (
        df.groupby(["mavaovien", "maxn"])["ketqua"]
        .mean()
        .unstack(level="maxn")
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # --- Visit-level demographics & vitals ---
    first_per_visit = df.groupby("mavaovien").first().reset_index()

    optional_vitals = ["tam_truong", "tam_thu", "mach"]
    extra_vital_cols = [c for c in optional_vitals if c in first_per_visit.columns]
    demo_cols = ["mavaovien", "tuoi", "phai"] + extra_vital_cols
    demo = first_per_visit[demo_cols].copy()

    # Encode phai
    phai_col = demo["phai"]
    if phai_col.dtype == object:
        phai_map = {
            "nam": 1, "Nam": 1, "NAM": 1, "male": 1, "Male": 1, "MALE": 1,
            "nữ": 0, "Nữ": 0, "NỮ": 0, "nu": 0, "Nu": 0,
            "female": 0, "Female": 0, "FEMALE": 0,
        }
        demo["phai"] = phai_col.map(phai_map).fillna(pd.to_numeric(phai_col, errors="coerce"))
    else:
        demo["phai"] = pd.to_numeric(phai_col, errors="coerce")

    demo["tuoi"] = pd.to_numeric(demo["tuoi"], errors="coerce")
    for vc in extra_vital_cols:
        demo[vc] = pd.to_numeric(demo[vc], errors="coerce")

    # --- Merge all ---
    result = visit_target.merge(demo, on="mavaovien", how="left")
    result = result.merge(pivot, on="mavaovien", how="left")
    result = result.drop(columns=["mavaovien"])

    # Ensure all columns are numeric
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Drop all-NaN and extremely sparse (>95% NaN) feature columns
    # Only target, tuoi, phai are fixed — all other cols (vitals + maxn) are filtered equally
    core_cols = ["target", "tuoi", "phai"] + extra_vital_cols
    feature_cols = [c for c in result.columns if c not in core_cols]
    all_nan_feats = [c for c in feature_cols if result[c].isna().all()]
    if all_nan_feats:
        result = result.drop(columns=all_nan_feats)
    feature_cols = [c for c in result.columns if c not in core_cols]
    sparse_feats = [c for c in feature_cols if result[c].isna().mean() > 1]
    if sparse_feats:
        result = result.drop(columns=sparse_feats)

    # Build maxn → tenxn mapping (only Format A has tenxn)
    if has_tenxn:
        maxn_to_tenxn = (
            df.dropna(subset=["maxn", "tenxn"])
            .groupby("maxn")["tenxn"]
            .first()
            .to_dict()
        )
    else:
        maxn_to_tenxn = {}

    return result, maxn_to_tenxn


def prepare_features_for_aco(df, target_col):
    """Return X (all features) and y for ACO to search over."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
