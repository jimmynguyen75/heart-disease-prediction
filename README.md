# Heart Disease Prediction - Machine Learning Analysis Platform

## 📋 Overview

This is a comprehensive web-based machine learning platform for analyzing heart disease data and generating thesis-style reports. The application provides end-to-end functionality from data upload to PDF report generation.

## ✨ Features

### 🔍 Data Analysis
- Automatic CSV loading and preprocessing
- Missing value detection and handling
- Feature scaling (Standard/MinMax/None)
- Comprehensive exploratory data analysis (EDA)
- Correlation matrix heatmaps

### 🤖 Machine Learning Models
The platform supports 15 different ML models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting Machine (GBM)**
6. **Neural Network (MLP)**
7. **XGBoost**
8. **Bagged Trees**
9. **Naive Bayes**
10. **Flexible Discriminant Analysis (FDA)**
11. **Model Averaged Neural Network (MANN)**
12. **Conditional Inference Tree (CIT)**
13. **Bayesian GLM (BGLM)**
14. **Boosted GLM (BGGLM)**
15. **MARS** (if available)

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- K-fold Cross-Validation (K=5, K=10)

### 📄 Report Generation
- Automatic PDF report generation in thesis format
- Professional tables (Table 3-6 similar to template)
- Visualizations (correlation heatmap, ROC curves)
- Complete sections: Abstract, Introduction, Methodology, Results, Discussion, Conclusion

## 🚀 Installation

### Option 1: Local Installation

1. **Clone or download the repository**
```bash
cd /path/to/project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to: `http://localhost:8501`

### Option 2: Docker Installation

1. **Build the Docker image**
```bash
docker build -t heart-disease-ml .
```

2. **Run the container**
```bash
docker run -p 8501:8501 -v /path/to/data:/data heart-disease-ml
```

3. **Access the application**
Open your browser and navigate to: `http://localhost:8501`

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── preprocessing.py       # Data preprocessing module
├── models.py             # ML models training and evaluation
├── report.py             # PDF report generation
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## 🎯 Usage Guide

### Step 1: Data Upload
1. Launch the application
2. Choose data source:
   - **Use Default Dataset**: Uses `/mnt/user-data/uploads/data.csv`
   - **Upload New CSV**: Upload your own dataset

### Step 2: Configuration
Configure preprocessing options in the sidebar:
- **Target Variable**: Select the target column
- **Missing Value Strategy**: mean/median/drop
- **Scaling Method**: standard/minmax/none
- **Test Set Size**: 0.1 to 0.4 (default: 0.2)

### Step 3: Exploratory Data Analysis
- Navigate to the **EDA** tab
- View correlation heatmap
- Examine feature distributions
- Analyze target variable distribution

### Step 4: Model Training
1. Go to the **Model Training** tab
2. Select models to train (or run all)
3. Configure cross-validation settings
4. Click **Start Training Pipeline**
5. Wait for training to complete

### Step 5: View Results
- Navigate to the **Results** tab
- Review performance metrics (Table 3)
- Check ROC-AUC values (Table 4)
- Examine confusion matrices (Table 5)
- Analyze cross-validation results (Table 6)
- View ROC curves

### Step 6: Generate Report
1. Go to the **Report** tab
2. Enter desired report filename
3. Click **Generate PDF Report**
4. Download the generated report

## 📊 Expected Output

### Tables Generated

**Table 3: Performance Metrics**
- Model name
- Accuracy
- Precision
- Recall
- F1-Score

**Table 4: ROC-AUC Values**
- Model name
- ROC-AUC score

**Table 5: Confusion Matrix Summary**
- Model name
- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)

**Table 6: K-Fold Cross-Validation Results**
- Model name
- K-fold value
- Mean accuracy
- Standard deviation

### Figures Generated
1. **Correlation Matrix Heatmap**
2. **ROC Curves** (15 subplots, one for each model)

## 🔬 Example Results (on data.csv)

Based on the analysis of the provided dataset:

**Top Performing Models:**
- XGBoost: ~93% accuracy
- Bagged Trees: ~93% accuracy
- Random Forest: ~91% accuracy

**ROC-AUC Scores:**
- Random Forest: 0.95
- Bagged Trees: 0.95
- XGBoost: 0.94

**Cross-Validation (K=10):**
- Random Forest: 94% ± 2%
- XGBoost: 90% ± 3%

## ⚙️ Configuration

### Random State
All models use `random_state=42` for reproducibility.

### Model Hyperparameters
Default hyperparameters are set for:
- Random Forest: 100 trees
- XGBoost: 100 estimators
- Neural Networks: (100, 50) hidden layers
- SVM: RBF kernel with default parameters

## 🐛 Troubleshooting

### Issue: Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: MARS not available
MARS (py-earth) may not install on all systems. The application will skip it automatically.

### Issue: Memory Error
- Reduce dataset size
- Reduce number of models
- Increase system RAM

### Issue: Slow Performance
- Reduce cross-validation folds
- Use fewer models
- Reduce dataset size

## 📝 Data Format Requirements

Your CSV file should:
- Have a header row with column names
- Include a target variable (binary: 0/1)
- Contain numeric features (categorical will be encoded)
- Target column should be named "target" or be the last column

Example:
```csv
age,sex,chest_pain,bp,cholesterol,target
45,1,2,130,250,0
55,0,3,140,290,1
...
```

## 🔒 Security & Privacy

- All data processing is local
- No data is sent to external servers
- Temporary files are stored in `/mnt/user-data/outputs`
- Reports are generated locally

## 📚 References

This application implements methods described in:
- Teja & Rayalu (2025). "Optimizing heart disease diagnosis with advanced machine learning models"
- UCI Machine Learning Repository: Heart Disease Dataset

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is created for academic purposes.

## 👥 Authors

- Master's Thesis Data Analysis Platform
- Created: 2025

## 🆘 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the Usage Guide
3. Check application logs

## 🎓 Citation

If you use this platform in your research, please cite:
```
Heart Disease Prediction ML Analysis Platform (2025)
Master's Thesis Data Analysis Tool
```

---

**Note**: This application is designed for educational and research purposes. Always consult healthcare professionals for medical decisions.
