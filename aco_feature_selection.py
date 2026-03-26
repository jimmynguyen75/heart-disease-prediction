"""
ACO Feature Selection Module (Lựa chọn đặc trưng bằng thuật toán Đàn Kiến)

Tham số theo luận văn:
  - 50 vòng lặp (iterations), 20 kiến mỗi vòng
  - alpha=1 (trọng số pheromone), beta=2 (trọng số heuristic)
  - rho=0.1 (tốc độ bay hơi pheromone), tau_init=4.0 (pheromone ban đầu)

Hàm fitness: AUC(S) - 0.05 * |S| / n_total_features
  → Tối đa hóa AUC đồng thời phạt nếu chọn quá nhiều features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# THAM SỐ ACO (Hyperparameters)
# =============================================================================

ACO_N_ANTS = 20       # Số kiến mỗi vòng lặp — nhiều kiến = khám phá rộng hơn
ACO_N_ITER = 50       # Số vòng lặp — nhiều vòng = hội tụ tốt hơn
ACO_ALPHA  = 1.0      # Trọng số pheromone (τ^alpha) — alpha=1: pheromone có ảnh hưởng tuyến tính
ACO_BETA   = 2.0      # Trọng số heuristic (η^beta)  — beta=2: heuristic quan trọng hơn pheromone
ACO_RHO    = 0.1      # Tốc độ bay hơi pheromone — 10% pheromone bị xóa mỗi vòng
                      # → Giúp "quên" kinh nghiệm cũ, tránh kẹt tại local optimum
ACO_TAU_INIT       = 4.0   # Pheromone khởi tạo — chọn = 4.0 để prob chọn ban đầu ≈ 50%
ACO_FITNESS_LAMBDA = 0.05  # Hệ số phạt số lượng features 
                            # Ví dụ: chọn 4/11 features → penalty = 0.05 * 4/11 = 0.018

# =============================================================================
# CÁC MODEL DÙNG ĐỂ SO SÁNH BỘ FEATURES (dùng trong compare_all_feature_sets)
# =============================================================================

COMPARISON_MODELS = {
    'Logistic Regression': lambda rs: LogisticRegression(random_state=rs, max_iter=1000),
    'Random Forest': lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
    'SVM': lambda rs: SVC(probability=True, random_state=rs),
    'KNN': lambda rs: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'GBM': lambda rs: GradientBoostingClassifier(n_estimators=100, random_state=rs),
    'Neural Network': lambda rs: MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs, max_iter=500),
    'XGBoost': lambda rs: xgb.XGBClassifier(n_estimators=100, random_state=rs, n_jobs=-1, eval_metric='logloss'),
    'Bagged Tree': lambda rs: BaggingClassifier(estimator=DecisionTreeClassifier(random_state=rs), n_estimators=100, random_state=rs, n_jobs=-1),
    'Naive Bayes': lambda _: GaussianNB(),
    'FDA': lambda _: LinearDiscriminantAnalysis(),
    'MANN': lambda rs: VotingClassifier(
        # 5 mạng neural với random_state khác nhau → mỗi mạng học theo hướng khác nhau
        # voting='soft': lấy trung bình xác suất của 5 mạng → ổn định hơn 1 mạng đơn lẻ
        estimators=[
            ('mlp1', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs,     max_iter=500)),
            ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 1, max_iter=500)),
            ('mlp3', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 2, max_iter=500)),
            ('mlp4', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 3, max_iter=500)),
            ('mlp5', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 4, max_iter=500)),
        ],
        voting='soft',
    ),
    'CIT': lambda rs: DecisionTreeClassifier(random_state=rs, criterion='entropy'),
}


# =============================================================================
# BƯỚC 1: TÍNH HEURISTIC ETA (thông tin tiên nghiệm về chất lượng từng feature)
# =============================================================================

def compute_heuristic(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Tính giá trị heuristic eta cho từng feature dựa trên Mutual Information (MI).

    MI đo lường mức độ "liên quan" giữa feature và biến mục tiêu (bệnh/không bệnh).
    Feature nào liên quan nhiều hơn → eta cao hơn → được ưu tiên chọn ngay từ đầu.

    Kết quả normalize về [0.1, 1.0] để tránh eta^beta = 0 (sẽ làm prob_select = 0 mãi mãi).

    Ví dụ kết quả (giả sử):
      "ST slope"         MI = 0.30 → eta = 0.95  (rất liên quan đến bệnh tim)
      "chest pain type"  MI = 0.22 → eta = 0.68
      "age"              MI = 0.14 → eta = 0.42
      "fasting blood sugar" MI = 0.02 → eta = 0.12  (ít liên quan)
    """
    # Tính mutual information giữa từng feature và target
    mi = mutual_info_classif(X, y, random_state=123)

    # Nếu MI = 0 (feature hoàn toàn không liên quan), gán giá trị nhỏ thay vì 0
    # để tránh score = 0 → prob_select = 0 → feature bị loại hoàn toàn ngay từ đầu
    mi = np.where(mi == 0, 1e-6, mi)

    # Normalize về [0.1, 1.0]:
    # - Giá trị nhỏ nhất → 0.1 (vẫn có cơ hội được chọn nhỏ)
    # - Giá trị lớn nhất → 1.0
    mi_min, mi_max = mi.min(), mi.max()
    if mi_max > mi_min:
        mi = 0.1 + 0.9 * (mi - mi_min) / (mi_max - mi_min)
    else:
        # Tất cả features có MI bằng nhau → gán 0.5 (trung bình)
        mi = np.full(len(mi), 0.5)
    return mi


# =============================================================================
# BƯỚC 2: HÀM FITNESS — Đánh giá chất lượng một tập features
# =============================================================================

def _aco_fitness(
    feature_indices: list,
    X: np.ndarray,
    y: np.ndarray,
    n_total_features: int,
    random_state: int = 123,
) -> float:
    """
    Tính điểm fitness cho một tập features S do một kiến chọn ra.

    Công thức: fitness(S) = AUC(S) - lambda * |S| / n_total
      - AUC(S): chất lượng phân loại khi dùng tập S (đánh giá bằng Logistic Regression)
      - lambda * |S| / n_total: phạt nếu chọn quá nhiều features

    Ví dụ:
      Kiến chọn được S = ["ST slope", "chest pain type", "age", "exercise angina"]
      → |S| = 4, n_total = 11
      → Train LR trên S → AUC = 0.91
      → penalty = 0.05 * 4/11 = 0.018
      → fitness = 0.91 - 0.018 = 0.892

      Nếu chọn thêm 1 feature thừa (không giúp ích):
      → AUC vẫn = 0.91, penalty = 0.05 * 5/11 = 0.023
      → fitness = 0.91 - 0.023 = 0.887  → thấp hơn → bị loại bỏ
    """
    # Nếu kiến không chọn feature nào → tập rỗng → fitness = 0 (vô dụng)
    if len(feature_indices) == 0:
        return 0.0

    # Lấy các cột feature mà kiến chọn
    X_sub = X[:, feature_indices]

    # Chia 80/20 để đánh giá nội bộ (stratify đảm bảo tỉ lệ bệnh/không bệnh đều nhau)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Scale trong nội bộ fitness (fit trên train, transform cả 2)
    # → LR cần scale để hội tụ tốt
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    try:
        # Dùng Logistic Regression vì: nhanh, đủ tốt để so sánh các tập features
        clf = LogisticRegression(max_iter=300, random_state=random_state)
        clf.fit(X_tr, y_tr)
        # predict_proba[:, 1] = xác suất thuộc class 1 (có bệnh)
        auc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    except Exception:
        # Trường hợp lỗi (ví dụ chỉ 1 class trong tập nhỏ) → fitness = 0
        return 0.0

    # Tính penalty: càng nhiều features → penalty càng lớn → khuyến khích chọn ít features
    penalty = ACO_FITNESS_LAMBDA * len(feature_indices) / n_total_features
    return float(auc - penalty)


# =============================================================================
# BƯỚC 3: MỖI KIẾN XÂY DỰNG TẬP FEATURES (ant construct solution)
# =============================================================================

def _ant_construct_solution(
    pheromone: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list:
    """
    Một con kiến quyết định chọn hay không chọn từng feature, hoàn toàn độc lập.

    Khác với ACO gốc (TSP) — nơi kiến chọn 1 trong N thành phố còn lại:
      Ở đây kiến xét từng feature và đưa ra quyết định nhị phân (include/exclude).

    Xác suất chọn feature j:
      score_j  = pheromone_j^alpha * eta_j^beta
      prob_j   = score_j / (score_j + mean_score)

    → Feature có score > trung bình: prob > 0.5 → thiên về được chọn
    → Feature có score < trung bình: prob < 0.5 → thiên về bị loại
    → Feature có score = trung bình: prob = 0.5 → hoàn toàn ngẫu nhiên

    Ví dụ (giả sử alpha=1, beta=2, pheromone đồng đều = 4.0):
      "ST slope":     score = 4.0^1 * 0.95^2 = 3.61
      "age":          score = 4.0^1 * 0.60^2 = 1.44
      "fasting BS":   score = 4.0^1 * 0.15^2 = 0.09
      mean_score = (3.61 + 1.44 + 0.09) / 3 = 1.71

      prob("ST slope")  = 3.61 / (3.61 + 1.71) = 0.68 → hay được chọn
      prob("age")       = 1.44 / (1.44 + 1.71) = 0.46 → gần 50/50
      prob("fasting BS")= 0.09 / (0.09 + 1.71) = 0.05 → gần như bị loại
    """
    n_features = len(pheromone)

    # Tính score cho từng feature: kết hợp pheromone (kinh nghiệm) và eta (chất lượng)
    # beta=2 → eta được bình phương → heuristic quan trọng hơn pheromone
    scores = (pheromone ** alpha) * (eta ** beta)

    # Ngưỡng so sánh = điểm trung bình toàn bộ features
    # → Feature "trung bình" sẽ có prob = 0.5
    mean_score = scores.mean()

    selected = []
    for j in range(n_features):
        # Xác suất chọn feature j
        prob_select = scores[j] / (scores[j] + mean_score)
        # Tung đồng xu có xác suất prob_select → chọn hay không
        if rng.random() < prob_select:
            selected.append(j)

    return selected  # Danh sách chỉ số (index) các features được chọn


# =============================================================================
# HÀM CHÍNH: CHẠY TOÀN BỘ THUẬT TOÁN ACO
# =============================================================================

def run_aco(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 123,
    progress_callback=None,
):
    """
    Chạy toàn bộ thuật toán ACO để tìm tập features tốt nhất.

    Luồng hoạt động tổng quát:
      1. Tính heuristic eta (1 lần, không đổi)
      2. Khởi tạo pheromone đồng đều = 4.0
      3. Lặp 50 vòng:
           a. 20 kiến mỗi vòng → mỗi kiến chọn 1 tập features
           b. Tính fitness cho từng tập
           c. Cập nhật pheromone: feature được chọn bởi kiến tốt → tăng
           d. Bay hơi 10% pheromone toàn bộ
      4. Trả về tập features có fitness cao nhất trong toàn bộ quá trình

    Parameters
    ----------
    X : DataFrame — toàn bộ features đầu vào (ví dụ: 11 features)
    y : Series — nhãn mục tiêu (0 = không bệnh, 1 = có bệnh)

    Returns
    -------
    selected_features : list[str] — tên các features được ACO chọn
    best_fitness      : float     — giá trị fitness tốt nhất đạt được
    history           : dict      — lịch sử fitness và số features qua từng vòng
    """
    # Khởi tạo bộ sinh số ngẫu nhiên (có seed để tái lập kết quả)
    rng = np.random.default_rng(random_state)
    feature_names = list(X.columns)
    n_features    = len(feature_names)
    X_arr = X.values   # Chuyển sang numpy array để tính toán nhanh hơn
    y_arr = y.values

    # --- BƯỚC 1: Tính heuristic eta (thực hiện 1 lần duy nhất) ---
    # eta phản ánh mức độ liên quan của từng feature với target
    # → Không thay đổi trong suốt quá trình chạy ACO
    eta = compute_heuristic(X, y)

    # --- BƯỚC 2: Khởi tạo pheromone ---
    # Tất cả features bắt đầu với pheromone = 4.0 (bằng nhau, chưa có kinh nghiệm)
    # Với tau=4.0 và eta trung bình ≈ 0.55: prob chọn ≈ 0.5 (50/50)
    pheromone = np.full(n_features, ACO_TAU_INIT)

    best_fitness = -np.inf
    best_subset  = list(range(n_features))  # Mặc định: chọn tất cả features (fallback)

    # Lưu lịch sử để vẽ đồ thị hội tụ
    history = {'best_fitness_per_iter': [], 'n_selected_per_iter': []}

    # --- BƯỚC 3: Vòng lặp chính (50 vòng) ---
    for iteration in range(ACO_N_ITER):
        iter_best_fitness = -np.inf
        iter_best_subset  = []
        # delta_pheromone: lượng pheromone sẽ được cộng thêm sau vòng này
        delta_pheromone = np.zeros(n_features)

        # --- 20 kiến hoạt động song song trong mỗi vòng ---
        for _ in range(ACO_N_ANTS):
            # Mỗi kiến tự xây dựng 1 tập features dựa trên pheromone + eta
            subset = _ant_construct_solution(pheromone, eta, ACO_ALPHA, ACO_BETA, rng)

            # Đánh giá chất lượng tập features vừa chọn
            fitness = _aco_fitness(subset, X_arr, y_arr, n_features, random_state=random_state)

            # Cộng pheromone cho các features trong tập này
            # Chia cho n_ants (20) để tránh pheromone tăng quá nhanh
            # → Feature được nhiều kiến tốt chọn → pheromone tăng nhiều hơn
            if fitness > 0 and len(subset) > 0:
                for j in subset:
                    delta_pheromone[j] += fitness / ACO_N_ANTS

            # Lưu kiến tốt nhất trong vòng này
            if fitness > iter_best_fitness:
                iter_best_fitness = fitness
                iter_best_subset  = subset

        # --- Cập nhật pheromone sau khi 20 kiến đã xong ---
        # Công thức: τ_new = (1 - ρ) * τ_old + Δτ
        #   (1 - 0.1) = 0.9: giữ lại 90% pheromone cũ (bay hơi 10%)
        #   Δτ: pheromone mới từ các kiến vừa đi
        pheromone = (1 - ACO_RHO) * pheromone + delta_pheromone

        # Giới hạn pheromone trong [0.5, 8.0]:
        #   - Tối thiểu 0.5: đảm bảo mọi feature vẫn có cơ hội được chọn
        #   - Tối đa 8.0: tránh 1 feature "độc quyền" hoàn toàn (pheromone explosion)
        pheromone = np.clip(pheromone, 0.5, 8.0)

        # Cập nhật kết quả tốt nhất toàn cục (qua tất cả các vòng)
        if iter_best_fitness > best_fitness and len(iter_best_subset) > 0:
            best_fitness = iter_best_fitness
            best_subset  = iter_best_subset

        # Ghi lịch sử để vẽ đồ thị hội tụ
        history['best_fitness_per_iter'].append(best_fitness)
        history['n_selected_per_iter'].append(len(best_subset))

        # Cập nhật progress bar trên giao diện (nếu có)
        if progress_callback is not None:
            progress_callback(iteration + 1, ACO_N_ITER)

    # Chuyển từ index → tên feature thực tế
    # Ví dụ: best_subset = [2, 4, 6, 9] → ["age", "chest pain type", "exercise angina", "ST slope"]
    selected_features = [feature_names[i] for i in best_subset]
    return selected_features, best_fitness, history


# =============================================================================
# HÀM PHỤ TRỢ: So sánh các bộ features (Baseline vs ACO vs All)
# =============================================================================

def _eval_feature_set(df_clean, target_col, feature_list, random_state=123):
    """
    Train tất cả COMPARISON_MODELS trên 1 bộ features và trả về metrics.
    Dùng để so sánh: Baseline features / ACO features / All features.
    """
    y = df_clean[target_col]
    X = df_clean[feature_list]

    # Split 80/20, stratify để giữ tỉ lệ bệnh/không bệnh
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Scale: fit trên train, transform cả train lẫn test
    # (không để thông tin test "rò rỉ" vào scaler)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}
    for name, builder in COMPARISON_MODELS.items():
        clf = builder(random_state)
        clf.fit(X_train_s, y_train)
        y_pred  = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]
        results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'roc_auc':  float(roc_auc_score(y_test, y_proba)),
            'f1':       float(f1_score(y_test, y_pred)),
        }
    return results


def compare_all_feature_sets(
    df_clean: pd.DataFrame,
    target_col: str,
    baseline_features: list,
    aco_features: list,
    random_state: int = 42,
):
    """
    Train toàn bộ models trên 3 bộ features để so sánh:
      1. Baseline: features theo paper gốc (6 features cố định)
      2. ACO:      features do ACO chọn (thường 4-7 features)
      3. All:      toàn bộ features có trong dataset

    Mục đích: Chứng minh ACO chọn được tập features nhỏ hơn nhưng kết quả
    tương đương hoặc tốt hơn so với dùng tất cả features.

    Returns: dict {label → {'results', 'n_features', 'feature_names'}}
    """
    all_features = [c for c in df_clean.columns if c != target_col]

    lbl_base = f'Baseline ({len(baseline_features)} features)'
    lbl_aco  = f'ACO ({len(aco_features)} features)'
    lbl_all  = f'All ({len(all_features)} features)'

    # Chạy All trước (cold start) để tránh JIT bias làm sai thời gian đo
    computed = {}
    for label, feats in [
        (lbl_all,  all_features),
        (lbl_base, baseline_features),
        (lbl_aco,  aco_features),
    ]:
        computed[label] = {
            'results':       _eval_feature_set(df_clean, target_col, feats, random_state),
            'n_features':    len(feats),
            'feature_names': feats,
        }

    # Trả về theo thứ tự hiển thị: Baseline → ACO → All
    return {k: computed[k] for k in [lbl_base, lbl_aco, lbl_all]}
