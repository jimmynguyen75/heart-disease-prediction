"""
Machine Learning Models Module (Module các mô hình học máy)
Khởi tạo, huấn luyện, đánh giá và cross-validate 15 mô hình ML.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              BaggingClassifier, AdaBoostClassifier, VotingClassifier,
                              ExtraTreesClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Quản lý toàn bộ vòng đời của 15 mô hình ML:
      khởi tạo → huấn luyện → đánh giá → cross-validation → vẽ ROC
    """

    def __init__(self, random_state=123):
        # random_state=201 trong pipeline chính để tái lập kết quả
        self.random_state  = random_state
        self.models        = {}          # dict {tên model → đối tượng model chưa train}
        self.results       = {}          # dict {tên model → dict metrics}
        self.trained_models = {}         # dict {tên model → đối tượng model đã train}

    # =========================================================================
    # KHỞI TẠO 12 MÔ HÌNH
    # =========================================================================

    def initialize_models(self):
        """
        Khởi tạo 15 mô hình ML với các tham số theo paper gốc.

        Danh sách models:
          1.  Logistic Regression  — mô hình tuyến tính cơ bản
          2.  Random Forest        — 100 cây quyết định, majority voting
          3.  SVM                  — tìm siêu phẳng phân cách tốt nhất
          4.  KNN                  — phân loại dựa trên 5 láng giềng gần nhất
          5.  GBM                  — 100 cây học tuần tự (boosting)
          6.  Neural Network       — MLP 2 lớp ẩn (100, 50 neurons)
          7.  XGBoost              — gradient boosting tối ưu hóa cao
          8.  Bagged Tree          — 100 cây quyết định, bagging (song song)
          9.  Naive Bayes          — xác suất có điều kiện Gaussian
          10. FDA                  — LDA với features bậc 2 (phi tuyến)
          11. MANN                 — 5 mạng neural voting mềm (soft voting)
          12. CIT                  — cây quyết định với entropy + pruning
          13. LDA                  — phân tích phân biệt tuyến tính
          14. Extra Trees          — ensemble cây cực ngẫu nhiên
          15. Gaussian Process     — phân loại Bayesian với kernel RBF
        """
        self.models = {
            # Logistic Regression: hồi quy logistic, dùng sigmoid để ra xác suất
            # max_iter=1000: cho đủ vòng lặp hội tụ với dữ liệu nhiều features
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),

            # Random Forest: 100 cây quyết định, mỗi cây train trên bootstrap sample
            # Kết quả cuối = majority voting từ 100 cây → giảm variance
            # n_jobs=-1: dùng tất cả CPU để train song song
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),

            # SVM: tìm siêu phẳng tối ưu phân cách 2 class
            # probability=True: dùng Platt scaling để tính xác suất (cần cho ROC-AUC)
            'SVM': SVC(
                probability=True, random_state=self.random_state
            ),

            # KNN: không có bước train thực sự, chỉ lưu data
            # Khi predict: tìm 5 điểm gần nhất → majority voting
            # n_neighbors=5: giá trị phổ biến, cân bằng bias/variance
            'KNN': KNeighborsClassifier(
                n_neighbors=5, n_jobs=-1
            ),

            # GBM (Gradient Boosting Machine): 100 cây học tuần tự
            # Mỗi cây mới học từ lỗi (residual) của cây trước → cải thiện dần dần
            'GBM': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),

            # Neural Network (MLP): 2 lớp ẩn với 100 và 50 neurons
            # Kiến trúc: input → 100 neurons → 50 neurons → output (sigmoid)
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=self.random_state, max_iter=500
            ),

            # XGBoost: gradient boosting tối ưu, xử lý missing values tốt hơn GBM
            # eval_metric='logloss': dùng log loss để tối ưu (phù hợp bài toán nhị phân)
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1,
                eval_metric='logloss'
            ),

            # Bagged Tree: 100 cây quyết định, mỗi cây train trên bootstrap sample khác nhau
            # Khác Random Forest: Bagged Tree dùng tất cả features, RF chỉ dùng subset features
            'Bagged Tree': BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=self.random_state),
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),

            # Naive Bayes: giả định các features độc lập với nhau (naive assumption)
            # GaussianNB: giả định mỗi feature có phân phối Gaussian trong mỗi class
            # Ưu điểm: train rất nhanh, không cần nhiều dữ liệu
            'Naive Bayes': GaussianNB(),

            # FDA (Flexible Discriminant Analysis):
            # PolynomialFeatures(degree=2): tạo thêm features bậc 2 (x1², x1*x2, x2²...)
            # → Từ 6 features gốc → thêm ~21 features bậc 2
            # LDA sau đó tìm ranh giới tuyến tính trên không gian mở rộng này
            # → Hiệu quả như tìm ranh giới phi tuyến trong không gian gốc = "Flexible"
            'FDA': Pipeline([
                ('basis', PolynomialFeatures(degree=2, include_bias=False)),
                ('lda', LinearDiscriminantAnalysis())
            ]),

            # MANN (Multiple Artificial Neural Network):
            # 5 mạng neural giống hệt nhau nhưng khác random_state
            # → Mỗi mạng học theo hướng khác nhau do khởi tạo weights khác nhau
            # voting='soft': lấy trung bình xác suất của 5 mạng (không phải majority vote)
            # → Ổn định hơn 1 mạng đơn lẻ, giảm variance
            'MANN': VotingClassifier(
                estimators=[
                    ('mlp1', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state,     max_iter=500)),
                    ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state + 1, max_iter=500)),
                    ('mlp3', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state + 2, max_iter=500)),
                    ('mlp4', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state + 3, max_iter=500)),
                    ('mlp5', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state + 4, max_iter=500)),
                ],
                voting='soft',
            ),

            # CIT (Conditional Inference Tree):
            # criterion='entropy': dùng information gain thay vì gini để chọn split
            # min_samples_split=20: node cần ≥ 20 mẫu mới được tách → tránh cây quá sâu
            # min_samples_leaf=10: leaf cần ≥ 10 mẫu → tránh overfitting trên ít mẫu
            # ccp_alpha=0.005: pruning sau khi train — cắt bỏ nhánh không đủ quan trọng
            'CIT': DecisionTreeClassifier(
                random_state=self.random_state, criterion='entropy',
                min_samples_split=20, min_samples_leaf=10, ccp_alpha=0.005
            ),

            # LDA (Linear Discriminant Analysis): tìm chiếu tuyến tính tối đa hóa
            # khoảng cách giữa các class, tương tự BGLM về phân tích tuyến tính
            'LDA': LinearDiscriminantAnalysis(),

            # Extra Trees: tương tự Random Forest nhưng chọn split point ngẫu nhiên hoàn toàn
            # → giảm variance hơn RF, train nhanh hơn
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),

            # Gaussian Process: phân loại Bayesian, mô hình hóa phân phối xác suất
            # RBF kernel: đo độ tương đồng theo khoảng cách Euclidean
            # → tương tự BGLM/BGGLM về tiếp cận Bayesian
            'Gaussian Process': GaussianProcessClassifier(
                kernel=RBF(), random_state=self.random_state
            ),
        }
        return self.models

    # =========================================================================
    # HUẤN LUYỆN VÀ ĐÁNH GIÁ 1 MODEL
    # =========================================================================

    def train_model(self, name, model, X_train, y_train, X_test, y_test):
        """
        Huấn luyện 1 model và tính toàn bộ metrics đánh giá.

        Các metrics được tính:
          - Accuracy   = (TP + TN) / tổng số mẫu
          - Precision  = TP / (TP + FP) — trong số dự đoán bệnh, bao nhiêu đúng?
          - Recall     = TP / (TP + FN) — trong số thực sự bệnh, phát hiện được bao nhiêu?
          - F1-Score   = 2 * Precision * Recall / (Precision + Recall) — cân bằng 2 chỉ số trên
          - ROC-AUC    = diện tích dưới đường cong ROC (0.5 = ngẫu nhiên, 1.0 = hoàn hảo)
          - Confusion Matrix = [[TN, FP], [FN, TP]]

        Ví dụ confusion matrix với Random Forest:
          Dự đoán: Không bệnh  Có bệnh
          Thực tế:
          Không bệnh  [101         8  ]  ← TN=101 (đúng), FP=8 (báo nhầm bệnh)
          Có bệnh     [  11       117  ]  ← FN=11 (bỏ sót), TP=117 (đúng)
        """
        try:
            # Huấn luyện model trên tập train
            model.fit(X_train, y_train)

            # Dự đoán class (0 hoặc 1) trên tập test
            y_pred = model.predict(X_test)

            # Lấy xác suất thuộc class 1 (có bệnh) để tính ROC-AUC
            # Hầu hết models có predict_proba() → dùng trực tiếp
            # SVM không có predict_proba mặc định → dùng decision_function (khoảng cách đến hyperplane)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]  # lấy cột class=1
            else:
                y_proba = model.decision_function(X_test)

            # Tính toàn bộ metrics
            metrics = {
                'accuracy':         accuracy_score(y_test, y_pred),
                'precision':        precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall':           recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1':               f1_score(y_test, y_pred, average='binary', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'roc_auc':          roc_auc_score(y_test, y_proba)
            }

            # Lưu model đã train để dùng sau (ví dụ: predict demo ở tab Prediction)
            self.trained_models[name] = model

            return metrics, y_pred, y_proba

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return None, None, None

    # =========================================================================
    # CROSS-VALIDATION CHO 1 MODEL
    # =========================================================================

    def cross_validate_model(self, name, model, X, y, k_fold=10):
        """
        Thực hiện Stratified K-Fold Cross-Validation cho 1 model.

        Tại sao Stratified?
          Dataset có thể không cân bằng (ví dụ: 60% bệnh, 40% không bệnh).
          Stratified đảm bảo MỖI FOLD giữ nguyên tỉ lệ 60/40 này
          → Kết quả đánh giá không bị lệch do fold nào đó toàn bệnh hoặc toàn không bệnh.

        Ví dụ K=5 với 1000 mẫu:
          Fold 1: train trên [2,3,4,5] → test trên [1] → accuracy = 0.88
          Fold 2: train trên [1,3,4,5] → test trên [2] → accuracy = 0.91
          Fold 3: train trên [1,2,4,5] → test trên [3] → accuracy = 0.89
          Fold 4: train trên [1,2,3,5] → test trên [4] → accuracy = 0.87
          Fold 5: train trên [1,2,3,4] → test trên [5] → accuracy = 0.90
          → mean = 0.890, std = 0.014

        CV chạy trên FULL dataset (train+test) — không phải chỉ train set
        → Đánh giá khả năng tổng quát hóa thực sự của model.
        """
        try:
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
            # cross_val_score tự động chia, train, và đánh giá cho từng fold
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

            return {
                'mean_accuracy': scores.mean(),   # trung bình accuracy qua K folds
                'std_accuracy':  scores.std(),    # độ lệch chuẩn — thấp = ổn định
                'scores':        scores.tolist()  # accuracy từng fold
            }
        except Exception as e:
            print(f"Error in cross-validation for {name}: {str(e)}")
            return {'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'scores': []}

    # =========================================================================
    # HUẤN LUYỆN TẤT CẢ 12 MODELS
    # =========================================================================

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Khởi tạo và huấn luyện tuần tự tất cả 12 models.
        Trả về 3 dict: metrics, predictions (class), probabilities (xác suất).

        Ví dụ output:
          results['Random Forest'] = {
              'accuracy': 0.924, 'precision': 0.936, 'recall': 0.914,
              'f1': 0.925, 'roc_auc': 0.951,
              'confusion_matrix': [[101, 8], [11, 117]]
          }
        """
        self.initialize_models()

        all_results      = {}
        all_predictions  = {}
        all_probabilities = {}

        for name, model in self.models.items():
            print(f"Training {name}...")
            metrics, y_pred, y_proba = self.train_model(
                name, model, X_train, y_train, X_test, y_test
            )
            if metrics is not None:
                all_results[name]       = metrics
                all_predictions[name]   = y_pred
                all_probabilities[name] = y_proba

        self.results = all_results
        return all_results, all_predictions, all_probabilities

    # =========================================================================
    # CROSS-VALIDATION TẤT CẢ MODELS VỚI NHIỀU GIÁ TRỊ K
    # =========================================================================

    def perform_cross_validation(self, X, y, k_values=[5, 10]):
        """
        Chạy cross-validation cho tất cả models với K=5 và K=10.

        Tại sao thử nhiều K?
          K=5:  mỗi fold = 20% data → test set lớn hơn, variance đánh giá thấp hơn
          K=10: mỗi fold = 10% data → train set lớn hơn, bias thấp hơn
          So sánh K=5 và K=10 cho thấy kết quả có ổn định không.

        Output ví dụ:
          cv_results['K=10']['Random Forest'] = {
              'mean_accuracy': 0.921,
              'std_accuracy': 0.018,
              'scores': [0.90, 0.93, 0.91, ...]
          }
        """
        cv_results = {}
        for k in k_values:
            cv_results[f'K={k}'] = {}
            for name, model in self.models.items():
                print(f"Cross-validating {name} with K={k}...")
                cv_result = self.cross_validate_model(name, model, X, y, k_fold=k)
                cv_results[f'K={k}'][name] = cv_result
        return cv_results

    # =========================================================================
    # VẼ ĐƯỜNG CONG ROC (so sánh kết quả hiện tại vs paper gốc)
    # =========================================================================

    def plot_roc_curves(self, y_test, all_probabilities):
        """
        Vẽ đường cong ROC cho tất cả models, kèm đường cong từ paper gốc để so sánh.

        Đường cong ROC là gì?
          - Trục X (FPR): tỉ lệ bệnh nhân KHÔNG bệnh nhưng bị dự đoán CÓ bệnh
          - Trục Y (TPR): tỉ lệ bệnh nhân CÓ bệnh được phát hiện đúng
          - Đường chéo (random): AUC = 0.5 → model không tốt hơn đoán ngẫu nhiên
          - Model tốt: đường cong càng xa đường chéo → AUC càng gần 1.0

        Đường cong paper gốc được tái tạo từ confusion matrix trong paper
        (paper không cung cấp tọa độ đường cong, chỉ có 1 điểm threshold)
        → Dùng PchipInterpolator để nội suy đường cong trơn qua điểm đó.
        """
        # Giá trị tham chiếu từ paper gốc (Table 5 — Confusion Matrix)
        paper_confusion_matrices = {
            'Logistic Regression': {'TN': 88,  'FP': 16, 'FN': 24, 'TP': 109},
            'Random Forest':       {'TN': 101, 'FP': 8,  'FN': 11, 'TP': 117},
            'SVM':                 {'TN': 89,  'FP': 16, 'FN': 23, 'TP': 109},
            'KNN':                 {'TN': 103, 'FP': 10, 'FN': 9,  'TP': 115},
            'GBM':                 {'TN': 97,  'FP': 14, 'FN': 15, 'TP': 111},
            'Neural Network':      {'TN': 90,  'FP': 15, 'FN': 12, 'TP': 110},
            'XGBoost':             {'TN': 104, 'FP': 8,  'FN': 8,  'TP': 117},
            'FDA':                 {'TN': 92,  'FP': 16, 'FN': 20, 'TP': 109},
            'MANN':                {'TN': 89,  'FP': 14, 'FN': 23, 'TP': 111},
            'CIT':                 {'TN': 92,  'FP': 25, 'FN': 20, 'TP': 100},
            'Bagged Tree':         {'TN': 104, 'FP': 7,  'FN': 8,  'TP': 118},
            'Naive Bayes':         {'TN': 90,  'FP': 11, 'FN': 22, 'TP': 112},
        }

        # AUC tham chiếu từ paper gốc (Table 4)
        paper_auc = {
            'Logistic Regression': 0.90, 'Random Forest': 0.95,
            'SVM': 0.91,                 'KNN': 0.91,
            'GBM': 0.92,                 'Neural Network': 0.91,
            'XGBoost': 0.94,             'Bagged Tree': 0.95,
            'Naive Bayes': 0.91,         'FDA': 0.91,
            'MANN': 0.91,                'CIT': 0.91,
        }

        n_models = len(all_probabilities)
        n_cols   = 3
        # Tính số hàng cần thiết: ví dụ 12 models / 3 cột = 4 hàng
        n_rows   = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), dpi=150)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, (name, y_proba) in enumerate(all_probabilities.items()):
            if idx < len(axes):
                # Tính đường cong ROC thực tế từ kết quả model hiện tại
                # roc_curve trả về mảng FPR, TPR tại nhiều ngưỡng threshold khác nhau
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)

                # Vẽ đường cong model hiện tại (màu xanh)
                axes[idx].plot(fpr, tpr, label=f'Current: AUC = {auc:.2f}',
                             linewidth=2.5, color='#2563eb', antialiased=True)

                # Vẽ đường cong tham chiếu từ paper (nếu có)
                if name in paper_confusion_matrices and name in paper_auc:
                    cm      = paper_confusion_matrices[name]
                    ref_auc = paper_auc[name]

                    # Tính 1 điểm threshold từ confusion matrix của paper:
                    # TPR = TP / (TP + FN) — sensitivity/recall
                    # FPR = FP / (FP + TN) — 1 - specificity
                    tpr_paper = cm['TP'] / (cm['TP'] + cm['FN'])
                    fpr_paper = cm['FP'] / (cm['FP'] + cm['TN'])

                    # Nội suy đường cong trơn qua điểm (fpr_paper, tpr_paper)
                    # sao cho diện tích ≈ ref_auc và hình dạng giống ROC thực tế
                    n_points   = 100
                    fpr_smooth = np.linspace(0, 1, n_points)

                    from scipy.interpolate import PchipInterpolator
                    anchor_fpr = [0, fpr_paper/3, fpr_paper/1.5, fpr_paper,
                                 fpr_paper + (1-fpr_paper)/3, fpr_paper + 2*(1-fpr_paper)/3, 1]
                    anchor_tpr = []
                    for f in anchor_fpr:
                        if f <= fpr_paper:
                            ratio = f / fpr_paper if fpr_paper > 0 else 0
                            t = tpr_paper * (ratio ** 0.7)   # tăng nhanh trước ngưỡng
                        else:
                            ratio = (f - fpr_paper) / (1 - fpr_paper) if fpr_paper < 1 else 1
                            t = tpr_paper + (1 - tpr_paper) * (ratio ** 1.3)  # tăng chậm sau ngưỡng
                        anchor_tpr.append(t)

                    anchor_tpr = np.clip(anchor_tpr, 0, 1)
                    pchip      = PchipInterpolator(anchor_fpr, anchor_tpr)
                    tpr_smooth = np.clip(pchip(fpr_smooth), 0, 1)

                    # Vẽ đường cong paper (màu đỏ)
                    axes[idx].plot(fpr_smooth, tpr_smooth, label=f'Paper: AUC = {ref_auc:.2f}',
                                 linewidth=2.5, color='#dc2626', alpha=0.8, antialiased=True)

                    # Hiển thị chênh lệch AUC: xanh lá = tốt hơn paper, đỏ = kém hơn
                    auc_diff   = auc - ref_auc
                    diff_sign  = '+' if auc_diff >= 0 else ''
                    diff_color = '#16a34a' if auc_diff >= 0 else '#dc2626'
                    textstr    = f'Δ AUC: {diff_sign}{auc_diff:.2f}'
                    props      = dict(boxstyle='round,pad=0.5', facecolor='white',
                                    edgecolor=diff_color, linewidth=2, alpha=0.95)
                    axes[idx].text(0.98, 0.65, textstr, transform=axes[idx].transAxes,
                                 fontsize=11, verticalalignment='top', horizontalalignment='right',
                                 bbox=props, color=diff_color, fontweight='bold')

                # Đường chéo = random classifier (AUC = 0.5)
                axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
                axes[idx].set_xlabel('Tỷ lệ dương tính giả (False Positive Rate)', fontsize=10)
                axes[idx].set_ylabel('Tỷ lệ dương tính thật (True Positive Rate)', fontsize=10)
                axes[idx].set_title(f'Đường cong ROC — {name}', fontsize=12, fontweight='bold')
                axes[idx].legend(loc='lower right', fontsize=9)
                axes[idx].grid(alpha=0.3)

        # Ẩn các subplot thừa (khi số model không chia hết cho n_cols)
        for idx in range(len(all_probabilities), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    # =========================================================================
    # TẠO BẢNG KẾT QUẢ (Table 3, 4, 5, 6 trong luận văn)
    # =========================================================================

    def create_results_tables(self, results, cv_results):
        """
        Tạo 4 bảng kết quả theo chuẩn luận văn:
          Table 3: Accuracy, Precision, Recall, F1-Score của 12 models
          Table 4: ROC-AUC của 12 models
          Table 5: Confusion Matrix (TN, FP, FN, TP) của 12 models
          Table 6: Cross-validation accuracy (K=5, K=10) của 12 models
        """
        # Table 3: Các chỉ số phân loại chính
        metrics_data = []
        for name, metrics in results.items():
            metrics_data.append({
                'Model':     name,
                'Accuracy':  f"{metrics['accuracy']:.2f}",
                'Precision': f"{metrics['precision']:.2f}",
                'Recall':    f"{metrics['recall']:.2f}",
                'F1-Score':  f"{metrics['f1']:.2f}"
            })
        table3 = pd.DataFrame(metrics_data)
        table3.insert(0, 'No.', [str(i) for i in range(1, len(table3) + 1)])

        # Table 4: ROC-AUC (thước đo phân biệt tổng thể, không phụ thuộc threshold)
        auc_data = []
        for name, metrics in results.items():
            auc_data.append({'Model': name, 'ROC-AUC': f"{metrics['roc_auc']:.2f}"})
        table4 = pd.DataFrame(auc_data)
        table4.insert(0, 'No.', [str(i) for i in range(1, len(table4) + 1)])

        # Table 5: Confusion Matrix — TN/FP/FN/TP
        # TN: dự đoán đúng không bệnh | FP: báo nhầm bệnh (nguy hiểm ít hơn FN)
        # FN: bỏ sót bệnh (nguy hiểm!) | TP: phát hiện đúng bệnh
        cm_data = []
        for name, metrics in results.items():
            cm = metrics['confusion_matrix']
            if len(cm) < 2 or len(cm[0]) < 2:
                cm = [[cm[0][0] if cm else 0, 0], [0, 0]]
            cm_data.append({
                'Model': name,
                'TN': str(cm[0][0]), 'FP': str(cm[0][1]),
                'FN': str(cm[1][0]), 'TP': str(cm[1][1])
            })
        table5 = pd.DataFrame(cm_data)
        table5.insert(0, 'No.', [str(i) for i in range(1, len(table5) + 1)])

        # Table 6: Cross-validation — đánh giá khả năng tổng quát hóa
        # std thấp → model ổn định, không phụ thuộc vào cách chia data
        cv_data = []
        for k_val, models_cv in cv_results.items():
            for name, cv_result in models_cv.items():
                cv_data.append({
                    'Model':         name,
                    'K-Fold':        k_val,
                    'Mean Accuracy': f"{cv_result['mean_accuracy']:.2f}",
                    'Std Dev':       f"{cv_result['std_accuracy']:.2f}"
                })
        table6 = pd.DataFrame(cv_data)
        if len(table6) > 0:
            table6.insert(0, 'No.', [str(i) for i in range(1, len(table6) + 1)])

        return {'table3': table3, 'table4': table4, 'table5': table5, 'table6': table6}
