"""
Main Streamlit Application for Heart Disease Prediction Analysis
Master's Thesis Data Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import DataPreprocessor, prepare_features_for_aco
from models import ModelTrainer
from report import ThesisReportGenerator
from aco_feature_selection import run_aco, compare_all_feature_sets, ACO_N_ITER
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease ML Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .header-container {
        background: linear-gradient(135deg, #0c1e3d 0%, #1e3a70 50%, #2563eb 100%);
        padding: 3rem 2.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .university-name {
        color: #ffffff !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.5px;
        line-height: 1.4;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .university-short {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0 0 1.5rem 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .project-title {
        color: #fbbf24 !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 1.5rem 0;
        padding-top: 1.5rem;
        border-top: 2px solid rgba(251,191,36,0.3);
        line-height: 1.5;
    }
    .info-row {
        display: flex;
        align-items: center;
        background: rgba(255,255,255,0.08);
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 4px solid #fbbf24;
        transition: all 0.2s ease;
    }
    .info-row:hover {
        background: rgba(255,255,255,0.12);
        border-left-color: #f59e0b;
        transform: translateX(3px);
    }
    .info-icon {
        font-size: 1.3rem;
        margin-right: 1rem;
        min-width: 30px;
        text-align: center;
    }
    .info-content {
        flex: 1;
    }
    .info-label {
        color: #fbbf24;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
        margin-bottom: 0.25rem;
    }
    .info-value {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        line-height: 1.4;
    }
    .logo-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 180px;
        border: 2px solid rgba(148,163,184,0.2);
    }
    /* Force all table columns to align left - including number columns */
    [data-testid="stDataFrame"] table td,
    [data-testid="stDataFrame"] table th {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] div[data-testid="StyledDataFrameRowHeader"],
    [data-testid="stDataFrame"] div[data-testid="StyledDataFrameDataCell"] {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {
        text-align: left !important;
    }
    /* Target number type columns specifically */
    [data-testid="stDataFrame"] [data-testid="column-header"],
    [data-testid="stDataFrame"] [data-testid="cell"] {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] .dataframe tbody tr th,
    [data-testid="stDataFrame"] .dataframe thead tr th {
        text-align: left !important;
    }
    /* Override any inline styles */
    [data-testid="stDataFrame"] * {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)

def _run_aco_selection(df_clean, target_col):
    """Run ACO and store selected features + convergence history in session_state."""
    X_all, y_all = prepare_features_for_aco(df_clean, target_col)

    progress_bar = st.progress(0, text=f"Running ACO (0/{ACO_N_ITER} iterations)...")

    def _progress(iter_num, n_iter):
        pct = int(iter_num / n_iter * 100)
        progress_bar.progress(pct, text=f"Running ACO ({iter_num}/{n_iter} iterations)...")

    aco_features, aco_fitness, aco_history = run_aco(
        X_all, y_all, random_state=123, progress_callback=_progress
    )
    progress_bar.progress(100, text="ACO complete!")

    st.session_state['aco_features'] = aco_features
    st.session_state['aco_fitness'] = aco_fitness
    st.session_state['aco_history'] = aco_history


def _show_feature_comparison(comparison: dict):
    """Display side-by-side comparison of Baseline / ACO / All feature sets."""
    set_labels = list(comparison.keys())
    model_names = list(next(iter(comparison.values()))['results'].keys())
    n_all = max(v['n_features'] for v in comparison.values())

    # --- Summary table (thesis-ready) ---
    st.markdown("### Summary")
    summary_rows = []
    for label in set_labels:
        meta = comparison[label]
        res = meta['results']
        n_feat = meta['n_features']
        reduction = (n_all - n_feat) / n_all * 100 if n_feat < n_all else 0.0
        avg_auc = np.mean([res[m]['roc_auc'] for m in model_names])
        avg_acc = np.mean([res[m]['accuracy'] for m in model_names])
        avg_f1 = np.mean([res[m]['f1'] for m in model_names])
        summary_rows.append({
            'Feature Set': label,
            '# Features': n_feat,
            'Reduction vs All': f"-{reduction:.1f}%" if reduction > 0 else "—",
            'Avg AUC': f"{avg_auc:.4f}",
            'Avg Accuracy': f"{avg_acc:.4f}",
            'Avg F1': f"{avg_f1:.4f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
    st.markdown("---")

    # --- Per-model tabs ---
    metrics = [('roc_auc', 'AUC'), ('accuracy', 'Accuracy'), ('f1', 'F1')]
    tab_auc, tab_acc, tab_f1 = st.tabs(["ROC-AUC", "Accuracy", "F1"])

    for tab, (metric_key, metric_label) in zip([tab_auc, tab_acc, tab_f1], metrics):
        with tab:
            rows = []
            for model in model_names:
                row = {'Model': model}
                for label in set_labels:
                    val = comparison[label]['results'][model][metric_key]
                    row[label] = f"{val:.4f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            x = np.arange(len(model_names))
            width = 0.25
            colors = ['steelblue', 'darkorange', 'seagreen']
            fig, ax = plt.subplots(figsize=(12, 4))
            for i, (label, color) in enumerate(zip(set_labels, colors)):
                vals = [comparison[label]['results'][m][metric_key] for m in model_names]
                bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
                ax.bar_label(bars, fmt='%.3f', fontsize=7, rotation=90, padding=2)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — Baseline vs ACO vs All Features")
            ax.set_xticks(x + width)
            ax.set_xticklabels(model_names, rotation=20, ha='right')
            ax.set_ylim(0, 1.15)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Recommendation (based on average AUC)
    avg_aucs = {
        label: np.mean([comparison[label]['results'][m]['roc_auc'] for m in model_names])
        for label in set_labels
    }
    best_label = max(avg_aucs, key=avg_aucs.get)
    st.markdown("**📌 Average AUC by feature set:**")
    for label, auc in avg_aucs.items():
        marker = " ✅ **(Best)**" if label == best_label else ""
        st.markdown(f"- **{label}**: {auc:.4f}{marker}")


def main():
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Default Dataset", "Upload New CSV"]
    )
    
    # Load data
    preprocessor = DataPreprocessor(random_state=123)
    
    if data_source == "Use Default Dataset":
        # Use local path instead of /mnt
        default_path = "data/data.csv"
        if not os.path.exists(default_path):
            # Try alternative path
            default_path = "/mnt/user-data/uploads/data.csv"

        if os.path.exists(default_path):
            df = preprocessor.load_data(default_path)
            st.sidebar.success(f"✅ Loaded default dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.sidebar.error("Default dataset not found! Please upload a CSV file instead.")
            return
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.sidebar.info("Please upload a CSV file")
            return
    
    # Target column selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Target Variable")
    detected_target = preprocessor.detect_target_column(df)
    target_col = st.sidebar.selectbox(
        "Select target column:",
        df.columns.tolist(),
        index=df.columns.tolist().index(detected_target) if detected_target in df.columns else 0
    )
    
    # Preprocessing options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Preprocessing Options")
    missing_strategy = st.sidebar.selectbox(
        "Missing value strategy:",
        ["mean", "median", "drop"]
    )
    
    scaling_method = st.sidebar.selectbox(
        "Scaling method:",
        ["standard", "minmax", "none"]
    )
    
    test_size = st.sidebar.slider(
        "Test set size:",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )
    
    # Cross-validation settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Cross-Validation")
    run_cv = st.sidebar.checkbox("Run K-fold CV", value=True)
    k_values = st.sidebar.multiselect(
        "Select K values:",
        [5, 10],
        default=[5, 10]
    )
    
    # Main content tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home",
        "📊 Data Overview",
        "🔍 EDA",
        "🤖 Model Training",
        "📈 Results",
        "📄 Report",
        "📚 Model Explanation",
        "🔮 Prediction Demo"
    ])

    # Tab 0: Home
    with tab0:
        if os.path.exists("logo.png"):
            import base64
            with open("logo.png", "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
                logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: 140px; height: auto;">'
        else:
            logo_html = '<div style="width: 140px; height: 140px; background: rgba(255,255,255,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-size: 3rem;">🏛️</div>'

        st.markdown(f"""
        <div class="header-container">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <div class="logo-container">
                    {logo_html}
                </div>
                <div style="flex: 1;">
                    <h1 class="university-name">HOCHIMINH CITY INTERNATIONAL UNIVERSITY</h1>
                    <p class="university-short">SCHOOL OF INFORMATION TECHNOLOGY</p>
                    <h2 class="project-title">❤️ Heart Disease Prediction - Machine Learning Analysis</h2>
                    <div class="info-row">
                        <div class="info-icon">👨‍🎓</div>
                        <div class="info-content">
                            <span class="info-label">Master's student</span>
                            <p class="info-value">Nguyen Minh Thu</p>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon">👨‍🏫</div>
                        <div class="info-content">
                            <span class="info-label">Instructor</span>
                            <p class="info-value">Assoc. Prof. Nguyen Thi Thuy Loan</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 1: Data Overview
    with tab1:
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Target Column", target_col)
        
        st.markdown("---")
        
        # Display first 100 rows
        st.subheader("Data Preview (First 100 rows)")
        st.dataframe(df.head(100))
        
        # Data summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Missing values
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Values Summary")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum() / len(df) * 100).values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            st.dataframe(missing_df)
    
    # Tab 2: EDA
    with tab2:
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix Heatmap")
        fig_corr = preprocessor.plot_correlation_heatmap(df)
        st.pyplot(fig_corr)
        
        # Target distribution
        st.subheader("Target Variable Distribution")
        fig_target, ax = plt.subplots(figsize=(8, 5))
        target_counts = df[target_col].value_counts().sort_index()
        labels = [str(v) for v in target_counts.index]
        ax.bar(labels, target_counts.values, color=['#1f4788', '#2c5aa0'])
        ax.set_xlabel(target_col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        st.pyplot(fig_target)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 0:
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_features:
                n_cols = 2
                n_rows = (len(selected_features) + 1) // 2
                fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
                
                for idx, col in enumerate(selected_features):
                    if idx < len(axes):
                        axes[idx].hist(df[col].dropna(), bins=30, color='#2c5aa0', edgecolor='black')
                        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Frequency')
                
                # Hide extra subplots
                for idx in range(len(selected_features), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_dist)
    
    # Tab 3: Model Training
    with tab3:
        st.markdown('<div class="sub-header">Machine Learning Model Training</div>', unsafe_allow_html=True)

        # --- Step 1: Feature Selection ---
        st.markdown("### Step 1: Choose Feature Set")
        feature_mode = st.radio(
            "Feature set to use for training:",
            ["Baseline", "ACO-selected features", "All features"],
            horizontal=True,
        )

        if feature_mode == "ACO-selected features":
            if st.button("🐜 Run ACO Feature Selection"):
                df_temp = preprocessor.handle_missing_values(df, strategy=missing_strategy)
                _run_aco_selection(df_temp, target_col)

            if 'aco_features' in st.session_state:
                aco_feats = st.session_state['aco_features']
                st.success(f"✅ ACO selected {len(aco_feats)} features: **{', '.join(aco_feats)}**")
                st.caption(f"Fitness score: {st.session_state['aco_fitness']:.4f}")

                # Convergence plot
                history = st.session_state['aco_history']
                fig_conv, ax_conv = plt.subplots(figsize=(7, 2.5))
                ax_conv.plot(range(1, len(history['best_fitness_per_iter']) + 1),
                             history['best_fitness_per_iter'], color='steelblue', linewidth=2)
                ax_conv.set_xlabel("Iteration")
                ax_conv.set_ylabel("Best Fitness")
                ax_conv.set_title("ACO Convergence")
                ax_conv.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_conv)
                plt.close(fig_conv)
            else:
                st.info("Click **Run ACO Feature Selection** to find the optimal feature subset.")

        st.markdown("---")

        # --- Step 2: Train Models ---
        st.markdown("### Step 2: Train Models")
        if st.button("🚀 Start Training Pipeline", type="primary"):
            # Determine feature list
            if feature_mode == "Baseline":
                feature_list = None  # split_data uses paper_features by default
            elif feature_mode == "ACO-selected features":
                if 'aco_features' not in st.session_state:
                    st.error("❌ Please run ACO Feature Selection first (Step 1).")
                    st.stop()
                feature_list = st.session_state['aco_features']
            else:  # All features
                feature_list = [c for c in df.columns if c != target_col]

            with st.spinner("Processing data..."):
                df_clean = preprocessor.handle_missing_values(df, strategy=missing_strategy)
                X_train, X_test, y_train, y_test = preprocessor.split_data(
                    df_clean, target_col, test_size=test_size,
                    scale_method=scaling_method, feature_list=feature_list
                )
                st.success(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)} | "
                           f"Features ({len(X_train.columns)}): {', '.join(X_train.columns.tolist())}")

            with st.spinner("Training models..."):
                trainer = ModelTrainer(random_state=123)
                results, predictions, probabilities = trainer.train_all_models(
                    X_train, y_train, X_test, y_test
                )
                st.success(f"✅ Trained {len(results)} models successfully!")


            # Cross-validation
            if run_cv and k_values:
                with st.spinner("Performing cross-validation..."):
                    X_full = pd.concat([X_train, X_test])
                    y_full = pd.concat([y_train, y_test])
                    cv_results = trainer.perform_cross_validation(X_full, y_full, k_values=k_values)
                    st.success(f"✅ Cross-validation complete for K={k_values}!")
            else:
                cv_results = {}

            # Store in session state
            st.session_state['results'] = results
            st.session_state['predictions'] = predictions
            st.session_state['probabilities'] = probabilities
            st.session_state['cv_results'] = cv_results
            st.session_state['y_test'] = y_test
            st.session_state['trainer'] = trainer
            st.session_state['preprocessor'] = preprocessor
            st.session_state['df'] = df_clean
            st.session_state['trained_feature_mode'] = feature_mode
            st.session_state['trained_features'] = list(X_train.columns)

            st.balloons()
            st.success("🎉 Training pipeline completed successfully!")


        # --- Feature Set Comparison ---
        st.markdown("---")
        st.markdown("### 📊 Compare All Feature Sets")
        st.caption("Train models on Baseline / ACO / All features side-by-side to find the best feature set.")

        if 'aco_features' not in st.session_state:
            st.info("🔔 Run **ACO Feature Selection** first (Step 1 → select ACO → click Run ACO).")
        else:
            if st.button("⚡ Run Feature Set Comparison"):
                paper_features = ['sex', 'chest pain type', 'fasting blood sugar',
                                  'resting ecg', 'exercise angina', 'ST slope']
                df_temp = preprocessor.handle_missing_values(df, strategy=missing_strategy)
                baseline_feats = [f for f in paper_features if f in df_temp.columns]
                aco_feats = st.session_state['aco_features']
                with st.spinner("Training all models on 3 feature sets... this may take a minute."):
                    comparison = compare_all_feature_sets(df_temp, target_col, baseline_feats, aco_feats, random_state=123)
                st.session_state['feature_comparison'] = comparison

            if 'feature_comparison' in st.session_state:
                _show_feature_comparison(st.session_state['feature_comparison'])

    # Tab 4: Results
    with tab4:
        st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.info("👈 Please run the training pipeline first in the 'Model Training' tab.")
        else:
            results = st.session_state['results']
            cv_results = st.session_state.get('cv_results', {})
            trainer = st.session_state['trainer']
            y_test = st.session_state['y_test']
            probabilities = st.session_state['probabilities']
            
            # Create results tables
            results_tables = trainer.create_results_tables(results, cv_results)
            
            # Display Table 3: Performance Metrics
            st.subheader("Table 3: Performance Metrics")
            st.dataframe(
                results_tables['table3'],
                hide_index=True,
                column_config={
                    "No.": st.column_config.NumberColumn(width="small"),
                    "Model": st.column_config.TextColumn(width="medium"),
                }
            )

            # Display Table 4: ROC-AUC
            st.subheader("Table 4: ROC-AUC Values")
            st.dataframe(
                results_tables['table4'],
                hide_index=True,
                column_config={
                    "No.": st.column_config.NumberColumn(width="small"),
                    "Model": st.column_config.TextColumn(width="medium"),
                }
            )

            # Display Table 5: Confusion Matrices
            st.subheader("Table 5: Confusion Matrix Summary")
            st.dataframe(
                results_tables['table5'],
                hide_index=True,
                column_config={
                    "No.": st.column_config.NumberColumn(width="small"),
                    "Model": st.column_config.TextColumn(width="medium"),
                }
            )

            # Display Table 6: Cross-validation
            if cv_results:
                st.subheader("Table 6: K-Fold Cross-Validation Results")
                st.dataframe(
                    results_tables['table6'],
                    hide_index=True,
                    column_config={
                        "No.": st.column_config.NumberColumn(width="small"),
                        "Model": st.column_config.TextColumn(width="medium"),
                    }
                )
            
            # ROC Curves
            st.subheader("Figure: ROC Curves for All Models")
            fig_roc = trainer.plot_roc_curves(y_test, probabilities)
            st.pyplot(fig_roc)

            # Model Comparison Chart
            st.markdown("---")
            st.subheader("Model Comparison Chart")

            # Prepare data for comparison
            model_names = list(results.keys())
            accuracies = [results[m]['accuracy'] for m in model_names]
            roc_aucs = [results[m]['roc_auc'] for m in model_names]
            f1_scores = [results[m]['f1'] for m in model_names]

            # Create comparison bar chart
            fig_compare, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Sort by accuracy for better visualization
            sorted_indices = np.argsort(accuracies)[::-1]
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_acc = [accuracies[i] for i in sorted_indices]
            sorted_auc = [roc_aucs[i] for i in sorted_indices]
            sorted_f1 = [f1_scores[i] for i in sorted_indices]

            # Color gradient based on performance
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))

            # Accuracy Chart
            bars1 = axes[0].barh(sorted_names, sorted_acc, color=colors)
            axes[0].set_xlabel('Accuracy', fontsize=12)
            axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0].set_xlim(0, 1)
            for i, (bar, val) in enumerate(zip(bars1, sorted_acc)):
                axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)

            # ROC-AUC Chart
            colors_auc = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
            bars2 = axes[1].barh(sorted_names, sorted_auc, color=colors_auc)
            axes[1].set_xlabel('ROC-AUC', fontsize=12)
            axes[1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
            axes[1].set_xlim(0, 1)
            for i, (bar, val) in enumerate(zip(bars2, sorted_auc)):
                axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)

            # F1-Score Chart
            bars3 = axes[2].barh(sorted_names, sorted_f1, color=colors)
            axes[2].set_xlabel('F1-Score', fontsize=12)
            axes[2].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
            axes[2].set_xlim(0, 1)
            for i, (bar, val) in enumerate(zip(bars3, sorted_f1)):
                axes[2].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_compare)

            # Best Model Summary
            best_acc_idx = np.argmax(accuracies)
            best_auc_idx = np.argmax(roc_aucs)
            best_f1_idx = np.argmax(f1_scores)

            st.markdown("#### Best Performing Models")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Accuracy", model_names[best_acc_idx], f"{accuracies[best_acc_idx]:.3f}")
            with col2:
                st.metric("Best ROC-AUC", model_names[best_auc_idx], f"{roc_aucs[best_auc_idx]:.3f}")
            with col3:
                st.metric("Best F1-Score", model_names[best_f1_idx], f"{f1_scores[best_f1_idx]:.3f}")

            # Feature Importance
            st.markdown("---")
            st.subheader("Feature Importance Analysis")
            st.write("Feature importance từ các models có hỗ trợ (Random Forest, XGBoost, GBM, etc.)")

            # Info expander explaining which models support feature importance
            with st.expander("ℹ️ Tại sao chỉ một số models có Feature Importance?"):
                st.markdown("""
                **Không phải tất cả ML models đều hỗ trợ Feature Importance.** Dưới đây là bảng giải thích:

                | Model | Hỗ trợ | Lý do |
                |-------|:------:|-------|
                | Random Forest | ✅ | Tree-based: tính từ Gini importance hoặc mean decrease impurity |
                | XGBoost | ✅ | Tree-based: tính từ gain, weight, hoặc cover |
                | GBM | ✅ | Tree-based: tính từ feature split frequency |
                | Bagged Tree | ✅ | Trung bình importance từ 100 Decision Trees |
                | CIT | ✅ | Decision Tree: tính từ information gain |
                | Logistic Regression | ❌ | Có `coef_` (hệ số) nhưng không phải importance trực tiếp |
                | SVM | ❌ | Kernel-based: không có feature importance (trừ linear kernel) |
                | KNN | ❌ | Instance-based: không học trọng số features |
                | Neural Network | ❌ | Black-box: weights phức tạp, khó interpret |
                | MANN | ❌ | Giống Neural Network |
                | Naive Bayes | ❌ | Probabilistic: dùng likelihood, không có importance |
                | FDA/LDA | ❌ | Có `coef_` nhưng là discriminant coefficients |

                **Lưu ý:**
                - `feature_importances_` là thuộc tính chuẩn của sklearn cho tree-based models
                - Các linear models (Logistic Regression, LDA) có thể dùng absolute coefficients như proxy
                - Neural Networks cần techniques đặc biệt như SHAP hoặc Permutation Importance
                """)


            # Get feature names from preprocessor
            if 'preprocessor' in st.session_state:
                preprocessor = st.session_state['preprocessor']
                if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names is not None:
                    feature_names = preprocessor.feature_names
                else:
                    # Try to get from df
                    df_stored = st.session_state.get('df', None)
                    if df_stored is not None:
                        feature_names = [col for col in df_stored.columns if col != target_col]
                    else:
                        feature_names = None
            else:
                feature_names = None

            # Models that support feature importance (must match exact names in models.py)
            importance_models = ['Random Forest', 'XGBoost', 'GBM', 'Bagged Tree', 'CIT']
            available_importance_models = []
            for m in importance_models:
                if m in trainer.trained_models:
                    model = trainer.trained_models[m]
                    # Check for direct feature_importances_ or estimators with it
                    if hasattr(model, 'feature_importances_'):
                        available_importance_models.append(m)
                    elif hasattr(model, 'estimators_'):
                        # BaggingClassifier: compute average from base estimators
                        if all(hasattr(est, 'feature_importances_') for est in model.estimators_):
                            available_importance_models.append(m)

            if available_importance_models and feature_names is not None:
                selected_importance_model = st.selectbox(
                    "Chọn model để xem Feature Importance:",
                    available_importance_models
                )

                if selected_importance_model:
                    model = trainer.trained_models[selected_importance_model]

                    # Get feature importances
                    importances = None
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'estimators_'):
                        # BaggingClassifier: average feature importance from all estimators
                        if all(hasattr(est, 'feature_importances_') for est in model.estimators_):
                            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

                    if importances is not None:

                        # Create importance dataframe
                        importance_df = pd.DataFrame({
                            'Feature': feature_names[:len(importances)],
                            'Importance': importances
                        }).sort_values('Importance', ascending=True)

                        # Plot
                        fig_imp, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
                        colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
                        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_imp)
                        ax.set_xlabel('Importance Score', fontsize=12)
                        ax.set_title(f'Feature Importance - {selected_importance_model}', fontsize=14, fontweight='bold')

                        # Add value labels
                        for bar, val in zip(bars, importance_df['Importance']):
                            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                                   f'{val:.4f}', va='center', fontsize=9)

                        plt.tight_layout()
                        st.pyplot(fig_imp)

                        # Top features table
                        st.markdown("#### Top 5 Most Important Features")
                        top_features = importance_df.nlargest(5, 'Importance')[['Feature', 'Importance']]
                        top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.4f}")
                        top_features = top_features.reset_index(drop=True)
                        top_features.insert(0, 'Rank', range(1, len(top_features) + 1))
                        st.dataframe(top_features, hide_index=True)

                        # Interpretation
                        st.markdown("#### Interpretation")
                        top_feature = importance_df.nlargest(1, 'Importance')['Feature'].values[0]
                        st.info(f"""
                        **Feature quan trọng nhất:** `{top_feature}`

                        Feature importance cho thấy mức độ đóng góp của mỗi feature vào việc dự đoán bệnh tim:
                        - **Giá trị cao**: Feature có ảnh hưởng lớn đến prediction
                        - **Giá trị thấp**: Feature ít ảnh hưởng đến prediction

                        Điều này giúp bác sĩ hiểu được yếu tố nào cần chú ý nhất khi đánh giá nguy cơ bệnh tim.
                        """)
                    else:
                        st.warning(f"Model {selected_importance_model} không hỗ trợ feature_importances_")
            else:
                if feature_names is None:
                    st.warning("Không thể lấy tên features. Hãy chạy training pipeline trước.")
                else:
                    st.warning("Không có model nào hỗ trợ feature importance được train.")

            # Combined Feature Importance (Average across models)
            if available_importance_models and feature_names is not None and len(available_importance_models) > 1:
                st.markdown("---")
                st.subheader("Combined Feature Importance (Average)")
                st.write("Trung bình feature importance từ tất cả các models hỗ trợ.")

                all_importances = []
                for model_name in available_importance_models:
                    model = trainer.trained_models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        all_importances.append(model.feature_importances_)
                    elif hasattr(model, 'estimators_'):
                        # BaggingClassifier
                        if all(hasattr(est, 'feature_importances_') for est in model.estimators_):
                            avg_imp = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                            all_importances.append(avg_imp)

                if all_importances:
                    avg_importance = np.mean(all_importances, axis=0)
                    std_importance = np.std(all_importances, axis=0)

                    combined_df = pd.DataFrame({
                        'Feature': feature_names[:len(avg_importance)],
                        'Avg Importance': avg_importance,
                        'Std': std_importance
                    }).sort_values('Avg Importance', ascending=True)

                    fig_combined, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
                    colors_combined = plt.cm.Oranges(np.linspace(0.4, 0.9, len(combined_df)))
                    bars = ax.barh(combined_df['Feature'], combined_df['Avg Importance'],
                                  xerr=combined_df['Std'], color=colors_combined, capsize=3)
                    ax.set_xlabel('Average Importance Score', fontsize=12)
                    ax.set_title(f'Combined Feature Importance (n={len(available_importance_models)} models)',
                               fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_combined)

    # Tab 5: Report
    with tab5:
        st.markdown('<div class="sub-header">Generate Thesis Report</div>', unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.info("👈 Please run the training pipeline first.")
        else:
            st.write("Generate a comprehensive PDF report in thesis format with all results.")
            
            report_name = st.text_input(
                "Report filename:",
                value="Thesis_Report.pdf"
            )
            
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Prepare data
                    results = st.session_state['results']
                    cv_results = st.session_state.get('cv_results', {})
                    trainer = st.session_state['trainer']
                    preprocessor = st.session_state['preprocessor']
                    df = st.session_state['df']
                    y_test = st.session_state['y_test']
                    probabilities = st.session_state['probabilities']
                    
                    # Create figures
                    fig_corr = preprocessor.plot_correlation_heatmap(df)
                    fig_roc = trainer.plot_roc_curves(y_test, probabilities)
                    
                    figures = {
                        'correlation_heatmap': fig_corr,
                        'roc_curves': fig_roc
                    }
                    
                    # Get data summary
                    data_summary = preprocessor.get_data_summary(df)
                    
                    # Create results tables
                    results_tables = trainer.create_results_tables(results, cv_results)
                    
                    # Generate report
                    # Use local outputs directory instead of /mnt
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, report_name)

                    report_gen = ThesisReportGenerator(output_path)
                    report_gen.generate_full_report(
                        data_summary, results_tables, figures, cv_results, output_path
                    )

                    st.success(f"✅ Report generated successfully!")
                    st.markdown(f"Report saved to: `{output_path}`")

                    # Provide download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=report_name,
                            mime="application/pdf"
                        )

    # Tab 6: Model Explanation
    with tab6:
        st.markdown('<div class="sub-header">Code Implementation - Machine Learning Models</div>', unsafe_allow_html=True)
        st.write("Chi tiết cách triển khai code cho từng model trong project này (file `models.py`).")

        # Model implementation details based on actual code
        model_implementations = {
            "Logistic Regression": {
                "sklearn_class": "LogisticRegression",
                "import_from": "sklearn.linear_model",
                "code": """LogisticRegression(
    random_state=123,
    max_iter=1000
)""",
                "parameters": {
                    "random_state=123": "Đảm bảo kết quả có thể tái tạo được (reproducibility)",
                    "max_iter=1000": "Số lần lặp tối đa để thuật toán hội tụ (mặc định là 100, tăng lên 1000 để đảm bảo hội tụ)"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Model sử dụng hàm sigmoid để chuyển đổi output thành xác suất (0-1)
2. Tối ưu hóa bằng thuật toán LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
3. `max_iter=1000` cho phép thuật toán có đủ thời gian hội tụ với dữ liệu phức tạp
4. Output: `predict_proba()` trả về xác suất, `predict()` trả về class (0 hoặc 1)
"""
            },
            "Random Forest": {
                "sklearn_class": "RandomForestClassifier",
                "import_from": "sklearn.ensemble",
                "code": """RandomForestClassifier(
    n_estimators=100,
    random_state=123,
    n_jobs=-1
)""",
                "parameters": {
                    "n_estimators=100": "Số lượng decision trees trong forest (100 cây)",
                    "random_state=123": "Seed cho random number generator để reproducibility",
                    "n_jobs=-1": "Sử dụng tất cả CPU cores để training song song (tăng tốc độ)"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Tạo 100 decision trees, mỗi cây được train trên một bootstrap sample khác nhau
2. Mỗi cây chỉ xem xét một subset ngẫu nhiên của features tại mỗi split
3. `n_jobs=-1` cho phép train tất cả 100 cây song song trên nhiều CPU cores
4. Prediction: Majority voting từ 100 cây để quyết định class cuối cùng
5. `predict_proba()`: Tỷ lệ số cây vote cho mỗi class
"""
            },
            "SVM": {
                "sklearn_class": "SVC",
                "import_from": "sklearn.svm",
                "code": """SVC(
    probability=True,
    random_state=123
)""",
                "parameters": {
                    "probability=True": "Cho phép tính xác suất prediction (cần cho ROC-AUC). Sử dụng Platt scaling",
                    "random_state=123": "Seed cho probability calibration"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Sử dụng kernel RBF (Radial Basis Function) mặc định
2. Tìm hyperplane tối ưu để phân tách 2 classes với margin lớn nhất
3. `probability=True`: Áp dụng Platt scaling để chuyển đổi distance thành probability
4. Lưu ý: Bật probability làm chậm training vì cần 5-fold CV nội bộ
5. Default C=1.0 (regularization parameter)
"""
            },
            "KNN": {
                "sklearn_class": "KNeighborsClassifier",
                "import_from": "sklearn.neighbors",
                "code": """KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)""",
                "parameters": {
                    "n_neighbors=5": "Số láng giềng gần nhất để vote (K=5)",
                    "n_jobs=-1": "Sử dụng tất cả CPU cores cho prediction"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Không có training phase thực sự - chỉ lưu trữ data
2. Khi predict: Tính khoảng cách Euclidean đến tất cả training points
3. Chọn 5 điểm gần nhất, majority voting để quyết định class
4. `predict_proba()`: Tỷ lệ số neighbors thuộc mỗi class (vd: 3/5 = 0.6)
5. K=5 là giá trị phổ biến, cân bằng giữa bias và variance
"""
            },
            "GBM": {
                "sklearn_class": "GradientBoostingClassifier",
                "import_from": "sklearn.ensemble",
                "code": """GradientBoostingClassifier(
    n_estimators=100,
    random_state=123
)""",
                "parameters": {
                    "n_estimators=100": "Số lượng boosting stages (100 cây tuần tự)",
                    "random_state=123": "Seed cho reproducibility"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Bắt đầu với prediction ban đầu (log-odds của class distribution)
2. Train 100 cây tuần tự, mỗi cây học từ residual errors của cây trước
3. Mỗi cây mới cố gắng sửa lỗi của ensemble hiện tại
4. Default learning_rate=0.1: Mỗi cây đóng góp 10% vào prediction
5. Default max_depth=3: Mỗi cây là "weak learner" với độ sâu thấp
6. Không hỗ trợ parallel training (tuần tự)
"""
            },
            "Neural Network": {
                "sklearn_class": "MLPClassifier",
                "import_from": "sklearn.neural_network",
                "code": """MLPClassifier(
    hidden_layer_sizes=(100, 50),
    random_state=123,
    max_iter=500
)""",
                "parameters": {
                    "hidden_layer_sizes=(100, 50)": "2 hidden layers: Layer 1 có 100 neurons, Layer 2 có 50 neurons",
                    "random_state=123": "Seed cho weight initialization",
                    "max_iter=500": "Số epochs tối đa để training"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Architecture: Input → 100 neurons → 50 neurons → Output (2 classes)
2. Default activation: ReLU cho hidden layers, Softmax cho output
3. Default optimizer: Adam (adaptive learning rate)
4. Default batch_size: 200 (mini-batch gradient descent)
5. Backpropagation để update weights qua 500 epochs (hoặc đến khi hội tụ)
6. Output: Softmax probabilities cho mỗi class
"""
            },
            "XGBoost": {
                "sklearn_class": "XGBClassifier",
                "import_from": "xgboost",
                "code": """xgb.XGBClassifier(
    n_estimators=100,
    random_state=123,
    n_jobs=-1,
    eval_metric='logloss'
)""",
                "parameters": {
                    "n_estimators=100": "Số lượng boosting rounds (100 cây)",
                    "random_state=123": "Seed cho reproducibility",
                    "n_jobs=-1": "Sử dụng tất cả CPU cores",
                    "eval_metric='logloss'": "Metric để đánh giá: Binary cross-entropy loss"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Gradient Boosting với regularization (L1 + L2)
2. Sử dụng second-order gradient (Hessian) để tối ưu tốt hơn
3. `n_jobs=-1`: Parallel tree construction (nhanh hơn sklearn GBM)
4. Tự động xử lý missing values
5. Default max_depth=6: Cây sâu hơn GBM nhưng có regularization
6. `eval_metric='logloss'`: Tối ưu cho binary classification
"""
            },
            "Bagged Tree": {
                "sklearn_class": "BaggingClassifier",
                "import_from": "sklearn.ensemble",
                "code": """BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=123),
    n_estimators=100,
    random_state=123,
    n_jobs=-1
)""",
                "parameters": {
                    "estimator=DecisionTreeClassifier()": "Base learner: Decision Tree với full depth",
                    "n_estimators=100": "Số lượng trees trong ensemble (100 cây)",
                    "random_state=123": "Seed cho bootstrap sampling",
                    "n_jobs=-1": "Parallel training trên tất cả CPU cores"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Tạo 100 bootstrap samples từ training data (sampling with replacement)
2. Train 1 Decision Tree trên mỗi bootstrap sample
3. Mỗi cây được train độc lập → có thể parallel
4. Khác với Random Forest: Sử dụng TẤT CẢ features tại mỗi split
5. Prediction: Majority voting từ 100 cây
6. Giảm variance nhưng không giảm bias
"""
            },
            "Naive Bayes": {
                "sklearn_class": "GaussianNB",
                "import_from": "sklearn.naive_bayes",
                "code": """GaussianNB()""",
                "parameters": {
                    "(no parameters)": "Sử dụng default settings, không cần hyperparameter tuning"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Giả định mỗi feature tuân theo phân phối Gaussian (Normal distribution)
2. Tính mean và variance của mỗi feature cho mỗi class
3. Áp dụng Bayes theorem: P(class|features) ∝ P(features|class) × P(class)
4. Giả định "naive": Các features độc lập với nhau given class
5. Training rất nhanh: Chỉ cần tính mean/variance
6. `predict_proba()`: Normalized posterior probabilities
"""
            },
            "FDA": {
                "sklearn_class": "LinearDiscriminantAnalysis",
                "import_from": "sklearn.discriminant_analysis",
                "code": """LinearDiscriminantAnalysis()""",
                "parameters": {
                    "(no parameters)": "Sử dụng default settings với SVD solver"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Tìm linear combinations của features tối đa hóa class separation
2. Giả định: Mỗi class có phân phối Gaussian với covariance matrix giống nhau
3. Tính between-class và within-class scatter matrices
4. Tìm discriminant functions để project data
5. Trong code này: Sử dụng LDA như proxy cho FDA (sklearn không có FDA riêng)
6. Default solver='svd': Singular Value Decomposition
"""
            },
            "MANN": {
                "sklearn_class": "MLPClassifier",
                "import_from": "sklearn.neural_network",
                "code": """MLPClassifier(
    hidden_layer_sizes=(100, 100, 50),
    random_state=123,
    max_iter=500
)""",
                "parameters": {
                    "hidden_layer_sizes=(100, 100, 50)": "3 hidden layers: 100 → 100 → 50 neurons (deeper network)",
                    "random_state=123": "Seed cho weight initialization",
                    "max_iter=500": "Số epochs tối đa"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Network sâu hơn Neural Network thường: 3 hidden layers thay vì 2
2. Architecture: Input → 100 → 100 → 50 → Output
3. Trong implementation này: Đây là single deeper MLP, không phải true ensemble
4. Mục đích: Capture more complex patterns với deeper architecture
5. Lưu ý: True MANN sẽ train nhiều networks và average predictions
6. Có thể prone to overfitting hơn do nhiều parameters hơn
"""
            },
            "CIT": {
                "sklearn_class": "DecisionTreeClassifier",
                "import_from": "sklearn.tree",
                "code": """DecisionTreeClassifier(
    random_state=123,
    criterion='entropy'
)""",
                "parameters": {
                    "random_state=123": "Seed cho reproducibility khi có ties",
                    "criterion='entropy'": "Sử dụng Information Gain (entropy) thay vì Gini impurity"
                },
                "explanation": """
**Cách hoạt động trong code:**
1. Xây dựng decision tree dựa trên Information Gain
2. `criterion='entropy'`: Split dựa trên việc giảm entropy (uncertainty)
3. Information Gain = Entropy(parent) - Weighted_Avg_Entropy(children)
4. Trong implementation này: Sử dụng Decision Tree với entropy như proxy cho CIT
5. True CIT sử dụng statistical tests (permutation tests) để chọn splits
6. sklearn không có native CIT implementation
"""
            }
        }

        # Create selection for model
        selected_model = st.selectbox(
            "Chọn model để xem chi tiết implementation:",
            list(model_implementations.keys())
        )

        if selected_model:
            impl = model_implementations[selected_model]

            # Display implementation details
            st.markdown(f"### {selected_model}")
            st.markdown(f"**Library:** `{impl['import_from']}`")
            st.markdown(f"**Class:** `{impl['sklearn_class']}`")

            # Code block
            st.markdown("#### Code Implementation")
            st.code(impl['code'], language='python')

            # Parameters explanation
            st.markdown("#### Giải thích Parameters")
            for param, desc in impl['parameters'].items():
                st.markdown(f"- **`{param}`**: {desc}")

            # How it works in this project
            st.markdown("#### Cách hoạt động trong Project")
            st.markdown(impl['explanation'])

            # Training and evaluation code
            st.markdown("#### Code Training & Evaluation")
            st.code("""# Training (trong models.py)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Get probabilities for ROC-AUC
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)""", language='python')

        # Summary table
        st.markdown("---")
        st.markdown("### Tổng hợp: Tất cả Models trong Project")

        summary_data = []
        for model_name, impl in model_implementations.items():
            # Extract key parameter
            params = list(impl['parameters'].keys())
            key_param = params[0] if params else "default"
            summary_data.append({
                "Model": model_name,
                "Class": impl['sklearn_class'],
                "Library": impl['import_from'].split('.')[-1],
                "Key Parameter": key_param
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True)

        # Cross-validation explanation
        st.markdown("---")
        st.markdown("### Cross-Validation Implementation")
        st.code("""# K-Fold Cross-Validation (trong models.py)
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=123)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

# Results
mean_accuracy = scores.mean()
std_accuracy = scores.std()""", language='python')

        st.markdown("""
**Giải thích:**
- `StratifiedKFold`: Đảm bảo mỗi fold có tỷ lệ class giống nhau như data gốc
- `shuffle=True`: Trộn data trước khi chia folds
- `n_jobs=-1`: Chạy parallel trên tất cả CPU cores
- K=5 và K=10: Hai giá trị K phổ biến để đánh giá model stability
""")

    # Tab 7: Prediction Demo
    with tab7:
        st.markdown('<div class="sub-header">Heart Disease Prediction Demo</div>', unsafe_allow_html=True)
        st.write("Nhập thông tin bệnh nhân để dự đoán nguy cơ bệnh tim từ tất cả models đã train.")

        if 'trainer' not in st.session_state or not st.session_state.get('results'):
            st.warning("⚠️ Vui lòng chạy Training Pipeline trước để sử dụng Prediction Demo.")
            st.info("👈 Đi đến tab 'Model Training' và click 'Start Training Pipeline'")
        else:
            trainer = st.session_state['trainer']
            df_stored = st.session_state.get('df', None)

            if df_stored is not None:
                # Determine feature_cols based on current feature_mode selection
                _paper_features = ['sex', 'chest pain type', 'fasting blood sugar',
                                   'resting ecg', 'exercise angina', 'ST slope']
                if feature_mode == "Baseline":
                    feature_cols = [f for f in _paper_features if f in df_stored.columns]
                elif feature_mode == "ACO-selected features":
                    feature_cols = st.session_state.get(
                        'aco_features',
                        [f for f in _paper_features if f in df_stored.columns]
                    )
                else:  # All features
                    feature_cols = [col for col in df_stored.columns if col != target_col]

                # Warn if selected feature set doesn't match what was trained
                trained_mode = st.session_state.get('trained_feature_mode')
                if trained_mode and trained_mode != feature_mode:
                    st.warning(f"⚠️ Form đang hiển thị features cho **{feature_mode}** "
                               f"nhưng model được train với **{trained_mode}**. "
                               f"Vui lòng **Train lại** trước khi dự đoán.")

                # Initialize sample_data in session state if not exists
                if 'sample_data' not in st.session_state:
                    st.session_state['sample_data'] = None
                if 'sample_info' not in st.session_state:
                    st.session_state['sample_info'] = None

                # Sample data selection
                st.markdown("### Load dữ liệu mẫu từ Dataset")
                st.write("Chọn một bệnh nhân mẫu từ dataset để test prediction:")

                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4, vertical_alignment="bottom")

                with col_btn1:
                    if st.button("🔴 Random (Có bệnh)", use_container_width=True):
                        positive_samples = df_stored[df_stored[target_col] == 1]
                        if len(positive_samples) > 0:
                            sample_row = positive_samples.sample(n=1, random_state=None).iloc[0]
                            for feature in feature_cols:
                                st.session_state[f"input_{feature}"] = float(sample_row[feature])
                            st.session_state['sample_info'] = "Có bệnh tim (target=1)"
                            st.rerun()

                with col_btn2:
                    if st.button("🟢 Random (Không bệnh)", use_container_width=True):
                        negative_samples = df_stored[df_stored[target_col] == 0]
                        if len(negative_samples) > 0:
                            sample_row = negative_samples.sample(n=1, random_state=None).iloc[0]
                            for feature in feature_cols:
                                st.session_state[f"input_{feature}"] = float(sample_row[feature])
                            st.session_state['sample_info'] = "Không có bệnh tim (target=0)"
                            st.rerun()

                with col_btn3:
                    st.caption(f"📋 Load theo index (0–{len(df_stored)-1})")
                    sub_input, sub_btn = st.columns([2, 1], vertical_alignment="bottom")
                    with sub_input:
                        sample_idx = st.number_input(
                            "index",
                            min_value=0,
                            max_value=len(df_stored)-1,
                            value=0,
                            key="sample_index_input",
                            help="Số thứ tự bệnh nhân trong dataset (0 = bệnh nhân đầu tiên)",
                            label_visibility="collapsed"
                        )
                    with sub_btn:
                        load_clicked = st.button("Load", use_container_width=True)

                with col_btn4:
                    reset_clicked = st.button("🔄 Reset", use_container_width=True)

                if load_clicked:
                    sample_row = df_stored.iloc[sample_idx]
                    for feature in feature_cols:
                        st.session_state[f"input_{feature}"] = float(sample_row[feature])
                    actual = "Có bệnh tim" if sample_row[target_col] == 1 else "Không có bệnh tim"
                    st.session_state['sample_info'] = f"Index {sample_idx} - {actual}"
                    st.rerun()

                if reset_clicked:
                    for feature in feature_cols:
                        if f"input_{feature}" in st.session_state:
                            del st.session_state[f"input_{feature}"]
                    st.session_state['sample_info'] = None
                    st.rerun()

                # Show current sample info
                if st.session_state.get('sample_info'):
                    st.success(f"✅ Đã load: **{st.session_state['sample_info']}**")

                st.markdown("---")
                st.markdown("### Nhập thông tin bệnh nhân")
                st.markdown("Điền các thông số dưới đây để dự đoán nguy cơ bệnh tim:")

                # Create input form with 3 columns
                input_values = {}
                cols = st.columns(3)

                for idx, feature in enumerate(feature_cols):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        min_val = float(df_stored[feature].min())
                        max_val = float(df_stored[feature].max())
                        mean_val = float(df_stored[feature].mean())
                        unique_vals = df_stored[feature].nunique()

                        if unique_vals <= 5:
                            options = sorted(df_stored[feature].unique().tolist())
                            # Set session_state default only if key absent (avoids conflict)
                            if f"input_{feature}" not in st.session_state:
                                st.session_state[f"input_{feature}"] = options[0]
                            input_values[feature] = st.selectbox(
                                f"{feature}",
                                options=options,
                                help=f"Giá trị: {options}",
                                key=f"input_{feature}"
                            )
                        else:
                            is_integer = df_stored[feature].dropna().apply(lambda x: x == int(x)).all()
                            if f"input_{feature}" not in st.session_state:
                                st.session_state[f"input_{feature}"] = int(mean_val) if is_integer else float(mean_val)
                            if is_integer:
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    min_value=int(min_val),
                                    max_value=int(max_val * 1.5),
                                    step=1,
                                    help=f"Range: {int(min_val)} - {int(max_val)}",
                                    key=f"input_{feature}"
                                )
                            else:
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    min_value=min_val,
                                    max_value=max_val * 1.5,
                                    step=(max_val - min_val) / 100,
                                    help=f"Range: {min_val:.2f} - {max_val:.2f}",
                                    key=f"input_{feature}"
                                )

                st.markdown("---")

                # Predict button
                if st.button("🔮 Predict Heart Disease Risk", type="primary"):
                    # Prepare input data
                    input_df = pd.DataFrame([input_values])

                    # Apply same preprocessing if scaler exists
                    if 'preprocessor' in st.session_state:
                        preprocessor = st.session_state['preprocessor']
                        if hasattr(preprocessor, 'scaler') and preprocessor.scaler is not None:
                            input_scaled = preprocessor.scaler.transform(input_df)
                            input_df = pd.DataFrame(input_scaled, columns=feature_cols)

                    st.markdown("### Prediction Results")

                    # Get predictions from all models
                    predictions = []
                    for model_name, model in trainer.trained_models.items():
                        try:
                            pred = model.predict(input_df)[0]
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_df)[0]
                                prob_positive = proba[1] if len(proba) > 1 else proba[0]
                            else:
                                prob_positive = pred

                            predictions.append({
                                'Model': model_name,
                                'Prediction': 'Có nguy cơ' if pred == 1 else 'Không có nguy cơ',
                                'Probability': prob_positive,
                                'Risk Level': 'High' if prob_positive > 0.7 else ('Medium' if prob_positive > 0.4 else 'Low')
                            })
                        except Exception:
                            predictions.append({
                                'Model': model_name,
                                'Prediction': 'Error',
                                'Probability': 0,
                                'Risk Level': 'N/A'
                            })

                    pred_df = pd.DataFrame(predictions)

                    # Summary metrics
                    positive_count = sum(1 for p in predictions if p['Prediction'] == 'Có nguy cơ')
                    avg_probability = np.mean([p['Probability'] for p in predictions if p['Prediction'] != 'Error'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Models dự đoán Có nguy cơ", f"{positive_count}/{len(predictions)}")
                    with col2:
                        st.metric("Xác suất trung bình", f"{avg_probability:.1%}")
                    with col3:
                        consensus = "Có nguy cơ" if positive_count > len(predictions) / 2 else "Không có nguy cơ"
                        st.metric("Consensus", consensus)

                    # Display results table
                    st.markdown("#### Chi tiết từng Model")

                    # Display table with formatted probability
                    display_df = pred_df.copy()
                    display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(display_df, hide_index=True)

                    # Visualization
                    st.markdown("#### Probability Comparison")
                    fig_pred, ax = plt.subplots(figsize=(10, 6))

                    colors_pred = ['#dc2626' if p['Prediction'] == 'Có nguy cơ' else '#16a34a'
                                  for p in predictions]

                    bars = ax.barh([p['Model'] for p in predictions],
                                  [p['Probability'] for p in predictions],
                                  color=colors_pred)

                    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold (0.5)')
                    ax.set_xlabel('Probability of Heart Disease', fontsize=12)
                    ax.set_title('Prediction Probability by Model', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 1)
                    ax.legend()

                    # Add value labels
                    for bar, prob in zip(bars, [p['Probability'] for p in predictions]):
                        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                               f'{prob:.1%}', va='center', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig_pred)

                    # Interpretation
                    st.markdown("#### Interpretation")
                    if consensus == "Có nguy cơ":
                        st.error(f"""
                        **Kết quả: Có nguy cơ bệnh tim**

                        - {positive_count}/{len(predictions)} models dự đoán bệnh nhân có nguy cơ bệnh tim
                        - Xác suất trung bình: {avg_probability:.1%}
                        - **Khuyến nghị:** Nên tham khảo ý kiến bác sĩ chuyên khoa tim mạch
                        """)
                    else:
                        st.success(f"""
                        **Kết quả: Không có nguy cơ cao**

                        - Chỉ {positive_count}/{len(predictions)} models dự đoán có nguy cơ
                        - Xác suất trung bình: {avg_probability:.1%}
                        - **Lưu ý:** Đây chỉ là dự đoán từ ML, không thay thế chẩn đoán y khoa
                        """)

            else:
                st.error("Không tìm thấy dữ liệu. Vui lòng load dataset trước.")

if __name__ == "__main__":
    main()
