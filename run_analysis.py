#!/usr/bin/env python3
"""
Standalone Script for Heart Disease ML Analysis
Run complete analysis pipeline from command line
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor
from models import ModelTrainer
from report import ThesisReportGenerator
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("Heart Disease ML Analysis - Standalone Pipeline")
    print("="*60)
    print()
    
    # Configuration
    DATA_PATH = "/mnt/user-data/uploads/data.csv"
    OUTPUT_DIR = "/mnt/user-data/outputs"
    REPORT_NAME = "Thesis_Report.pdf"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load Data
    print("Step 1: Loading data...")
    preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    df = preprocessor.load_data(DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Step 2: Data Summary
    print("Step 2: Generating data summary...")
    summary = preprocessor.get_data_summary(df)
    print(f"   Columns: {', '.join(summary['columns'])}")
    print(f"   Missing values: {sum(summary['missing_values'].values())}")
    print()
    
    # Step 3: Detect target
    print("Step 3: Detecting target variable...")
    target_col = preprocessor.detect_target_column(df)
    print(f"✅ Target column: {target_col}")
    print()
    
    # Step 4: Handle missing values
    print("Step 4: Preprocessing data...")
    df_clean = preprocessor.handle_missing_values(df, strategy='mean')
    print("✅ Missing values handled")
    print()
    
    # Step 5: Split data
    print("Step 5: Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_clean, target_col, test_size=TEST_SIZE, scale_method='standard'
    )
    print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()
    
    # Step 6: Generate correlation heatmap
    print("Step 6: Generating correlation heatmap...")
    fig_corr = preprocessor.plot_correlation_heatmap(df_clean)
    corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig_corr.savefig(corr_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {corr_path}")
    plt.close(fig_corr)
    print()
    
    # Step 7: Train models
    print("Step 7: Training machine learning models...")
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    results, predictions, probabilities = trainer.train_all_models(
        X_train, y_train, X_test, y_test
    )
    print(f"✅ Trained {len(results)} models successfully")
    print()
    
    # Display results
    print("Model Performance Summary:")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:30} | Acc: {metrics['accuracy']:.3f} | AUC: {metrics['roc_auc']:.3f}")
    print()
    
    # Step 8: Cross-validation
    print("Step 8: Performing K-fold cross-validation...")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    cv_results = trainer.perform_cross_validation(X_full, y_full, k_values=[5, 10])
    print("✅ Cross-validation completed")
    print()
    
    # Display CV results
    print("Cross-Validation Results (K=10):")
    print("-" * 60)
    for name, cv_result in cv_results.get('K=10', {}).items():
        print(f"{name:30} | Accuracy: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
    print()
    
    # Step 9: Generate ROC curves
    print("Step 9: Generating ROC curves...")
    fig_roc = trainer.plot_roc_curves(y_test, probabilities)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig_roc.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {roc_path}")
    plt.close(fig_roc)
    print()
    
    # Step 10: Create results tables
    print("Step 10: Creating results tables...")
    results_tables = trainer.create_results_tables(results, cv_results)
    
    # Save tables as CSV
    for table_name, table_df in results_tables.items():
        csv_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
        table_df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")
    print()
    
    # Display Table 3
    print("Table 3: Performance Metrics")
    print(results_tables['table3'].to_string(index=False))
    print()
    
    # Display Table 4
    print("Table 4: ROC-AUC Values")
    print(results_tables['table4'].to_string(index=False))
    print()
    
    # Step 11: Generate PDF report
    print("Step 11: Generating PDF report...")
    report_path = os.path.join(OUTPUT_DIR, REPORT_NAME)
    
    figures = {
        'correlation_heatmap': fig_corr,
        'roc_curves': fig_roc
    }
    
    report_gen = ThesisReportGenerator(report_path)
    
    # Recreate figures for report
    fig_corr = preprocessor.plot_correlation_heatmap(df_clean)
    fig_roc = trainer.plot_roc_curves(y_test, probabilities)
    figures = {
        'correlation_heatmap': fig_corr,
        'roc_curves': fig_roc
    }
    
    report_gen.generate_full_report(
        summary, results_tables, figures, cv_results, report_path
    )
    
    plt.close('all')
    
    print(f"✅ Report saved: {report_path}")
    print()
    
    # Summary
    print("="*60)
    print("Analysis Complete! 🎉")
    print("="*60)
    print()
    print("Generated Files:")
    print(f"  📊 Correlation Heatmap: {os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')}")
    print(f"  📈 ROC Curves: {os.path.join(OUTPUT_DIR, 'roc_curves.png')}")
    print(f"  📄 Table 3 (Metrics): {os.path.join(OUTPUT_DIR, 'table3.csv')}")
    print(f"  📄 Table 4 (AUC): {os.path.join(OUTPUT_DIR, 'table4.csv')}")
    print(f"  📄 Table 5 (Confusion): {os.path.join(OUTPUT_DIR, 'table5.csv')}")
    print(f"  📄 Table 6 (CV Results): {os.path.join(OUTPUT_DIR, 'table6.csv')}")
    print(f"  📑 PDF Report: {report_path}")
    print()
    print("✅ All outputs saved to: /mnt/user-data/outputs/")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
