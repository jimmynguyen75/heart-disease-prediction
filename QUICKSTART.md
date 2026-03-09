# Quick Start Guide - Heart Disease ML Analysis Platform

## 📦 Files Included

1. **app.py** - Main Streamlit application
2. **preprocessing.py** - Data preprocessing module  
3. **models.py** - Machine learning models (15 models)
4. **report.py** - PDF report generation
5. **run_analysis.py** - Standalone CLI script
6. **requirements.txt** - Python dependencies
7. **Dockerfile** - Docker configuration
8. **run_app.sh** - Bash script to launch app
9. **README.md** - Complete documentation

## 🚀 Quick Start

### Method 1: Run Streamlit Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or use the provided script
chmod +x run_app.sh
./run_app.sh
```

The app will be available at: http://localhost:8501

### Method 2: Run Standalone Analysis (Command Line)

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python run_analysis.py
```

This will:
- Load data from `/mnt/user-data/uploads/data.csv`
- Run all 15 ML models
- Perform K-fold cross-validation (K=5, K=10)
- Generate all tables (Table 3-6)
- Create figures (correlation heatmap, ROC curves)
- Generate PDF report: `Thesis_Report.pdf`

All outputs will be saved to: `/mnt/user-data/outputs/`

### Method 3: Docker

```bash
# Build image
docker build -t heart-disease-ml .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/data heart-disease-ml
```

## 📊 Expected Outputs

When you run the analysis, you will get:

### Files Generated:
1. `Thesis_Report.pdf` - Complete thesis-style report
2. `table3.csv` - Performance metrics (Accuracy, Precision, Recall, F1)
3. `table4.csv` - ROC-AUC values for all models
4. `table5.csv` - Confusion matrices summary
5. `table6.csv` - K-fold cross-validation results
6. `correlation_heatmap.png` - Feature correlation visualization
7. `roc_curves.png` - ROC curves for all 15 models

### Tables Format:

**Table 3: Performance Metrics**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.83 | 0.84 | 0.78 | 0.81 |
| Random Forest | 0.91 | 0.92 | 0.90 | 0.91 |
| XGBoost | 0.93 | 0.92 | 0.92 | 0.92 |
| ... | ... | ... | ... | ... |

**Table 4: ROC-AUC Values**
| Model | ROC-AUC |
|-------|---------|
| Random Forest | 0.95 |
| XGBoost | 0.94 |
| Bagged Tree | 0.95 |
| ... | ... |

**Table 6: Cross-Validation Results**
| Model | K-Fold | Mean Accuracy | Std Dev |
|-------|--------|---------------|---------|
| Random Forest | K=10 | 0.94 | 0.02 |
| XGBoost | K=10 | 0.90 | 0.03 |
| ... | ... | ... | ... |

## 🎯 Using the Web App

1. **Upload Data**: Choose default dataset or upload CSV
2. **Configure**: Set target variable, preprocessing options
3. **EDA Tab**: View data analysis and visualizations
4. **Model Training Tab**: Click "Start Training Pipeline"
5. **Results Tab**: View all tables and metrics
6. **Report Tab**: Generate PDF report

## 💡 Tips

- Default dataset is at: `/mnt/user-data/uploads/data.csv`
- All outputs save to: `/mnt/user-data/outputs/`
- Use `random_state=42` for reproducibility
- Report generation includes all tables and figures
- 15 models are trained automatically

## 🔧 Customization

Edit these parameters in the code:

```python
# In run_analysis.py or app.py
RANDOM_STATE = 42          # For reproducibility
TEST_SIZE = 0.2            # Train-test split ratio
K_VALUES = [5, 10]         # Cross-validation folds
```

## 📚 Models Included

1. Logistic Regression
2. Random Forest
3. SVM
4. KNN
5. GBM (Gradient Boosting)
6. Neural Network (MLP)
7. XGBoost
8. Bagged Trees
9. Naive Bayes
10. FDA (Flexible Discriminant Analysis)
11. MANN (Model Averaged Neural Network)
12. CIT (Conditional Inference Tree)
13. BGLM (Bayesian GLM)
14. BGGLM (Boosted GLM)
15. MARS (if available)

## 🆘 Troubleshooting

**Issue**: Dependencies fail to install
```bash
pip install --upgrade pip
pip install --break-system-packages -r requirements.txt
```

**Issue**: Data file not found
- Check path: `/mnt/user-data/uploads/data.csv`
- Or upload your own CSV in the app

**Issue**: Out of memory
- Reduce dataset size
- Train fewer models
- Reduce cross-validation folds

## 📞 Support

For detailed documentation, see README.md

---
Created for Master's Thesis Data Analysis
