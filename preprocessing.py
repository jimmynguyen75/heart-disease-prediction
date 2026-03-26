"""
Data Preprocessing Module (Module tiền xử lý dữ liệu)
Xử lý nạp dữ liệu, làm sạch, xử lý missing values, scale và chia tập train/test.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """
    Lớp tiền xử lý dữ liệu chính — dùng cho các dataset dạng wide (chuẩn tabular ML).
    Dataset dạng long của bệnh viện VN (_vn files) dùng hàm preprocess_vn_data() riêng bên dưới.
    """

    def __init__(self, random_state=123):
        # random_state đảm bảo kết quả tái lập (giống set.seed trong R)
        self.random_state  = random_state
        self.scaler        = None   # lưu scaler đã fit để dùng lại khi predict
        self.feature_names = None   # tên các features đã dùng để train
        self.target_name   = None   # tên cột target

    # =========================================================================
    # NẠP DỮ LIỆU
    # =========================================================================

    def load_data(self, file_path):
        """Đọc file CSV và trả về DataFrame."""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    # =========================================================================
    # THỐNG KÊ MÔ TẢ
    # =========================================================================

    def get_data_summary(self, df):
        """Trả về dict tóm tắt dataset: shape, dtypes, missing values, statistics."""
        summary = {
            'shape':               df.shape,
            'columns':             list(df.columns),
            'dtypes':              df.dtypes.to_dict(),
            'missing_values':      df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'statistics':          df.describe().to_dict()
        }
        return summary

    # =========================================================================
    # XỬ LÝ MISSING VALUES
    # =========================================================================

    def handle_missing_values(self, df, strategy='mean'):
        """
        Xử lý giá trị thiếu (missing values) theo 4 chiến lược:

        1. 'mean'   — thay bằng giá trị trung bình của cột
           Ví dụ: cholesterol = [200, NaN, 240] → [200, 213, 240]
           Phù hợp khi: phân phối đối xứng, không có outliers lớn

        2. 'median' — thay bằng giá trị trung vị
           Ví dụ: cholesterol = [200, NaN, 600] → [200, 200, 600]  (robust hơn mean khi có outlier)
           Phù hợp khi: phân phối lệch (skewed), có outliers

        3. 'knn'    — KNN Imputer: tìm 5 hàng gần nhất (theo Euclidean) và lấy trung bình
           Ví dụ: bệnh nhân 45 tuổi nam, thiếu cholesterol
                  → tìm 5 bệnh nhân nam ~45 tuổi → lấy trung bình cholesterol của họ
           Phù hợp khi: dữ liệu có cấu trúc, missing không ngẫu nhiên

        4. 'drop'   — xóa hàng có missing values
           Chỉ dùng khi số hàng bị xóa ít, không ảnh hưởng đến dataset.
        """
        df_clean = df.copy()

        # --- Chiến lược KNN (phức tạp nhất, xử lý riêng) ---
        if strategy == 'knn':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

            # KNNImputer không xử lý được cột toàn NaN → bỏ trước
            all_nan_cols = [c for c in numeric_cols if df_clean[c].isna().all()]
            if all_nan_cols:
                df_clean     = df_clean.drop(columns=all_nan_cols)
                numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]

            # Impute tất cả cột số cùng lúc (KNN xét tất cả features để tìm láng giềng)
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

            # Cột text/categorical (nếu có) → dùng mode (giá trị xuất hiện nhiều nhất)
            for col in df_clean.select_dtypes(exclude=[np.number]).columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_vals = df_clean[col].mode()
                    if len(mode_vals) > 0:
                        df_clean[col].fillna(mode_vals[0], inplace=True)
            return df_clean

        # --- Chiến lược mean / median / drop ---
        cols_to_drop = []
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    col_mean = df_clean[col].mean()
                    if pd.isna(col_mean):
                        # Cột toàn NaN → không tính được mean/median → bỏ cột
                        cols_to_drop.append(col)
                    elif strategy == 'mean':
                        df_clean[col].fillna(col_mean, inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif strategy == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                else:
                    # Cột categorical: thay bằng mode (giá trị phổ biến nhất)
                    mode_vals = df_clean[col].mode()
                    if len(mode_vals) > 0:
                        df_clean[col].fillna(mode_vals[0], inplace=True)

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        return df_clean

    # =========================================================================
    # TỰ ĐỘNG PHÁT HIỆN CỘT TARGET
    # =========================================================================

    def detect_target_column(self, df):
        """
        Tự động tìm cột target dựa trên tên phổ biến.
        Nếu không tìm được → dùng cột cuối cùng (quy ước thông thường của ML datasets).

        Ví dụ:
          Columns: ['age', 'sex', 'cholesterol', 'target'] → trả về 'target'
          Columns: ['age', 'sex', 'cholesterol', 'HeartDisease'] → trả về 'HeartDisease' (cột cuối)
        """
        target_keywords = ['target', 'Target', 'label', 'Label', 'class', 'Class']
        for col in df.columns:
            if col in target_keywords:
                return col
        return df.columns[-1]

    # =========================================================================
    # MA TRẬN TƯƠNG QUAN
    # =========================================================================

    def get_correlation_matrix(self, df):
        """Tính ma trận tương quan Pearson cho tất cả cột số."""
        numeric_df   = df.select_dtypes(include=[np.number])
        corr_matrix  = numeric_df.corr()
        return corr_matrix

    def plot_correlation_heatmap(self, df):
        """Vẽ heatmap ma trận tương quan. Màu đỏ = tương quan thuận, xanh = nghịch."""
        corr_matrix = self.get_correlation_matrix(df)
        fig, ax     = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax)
        plt.title('Ma trận tương quan (Correlation Matrix)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    # =========================================================================
    # CHIA TẬP TRAIN / TEST VÀ SCALE
    # =========================================================================

    def split_data(self, df, target_col, test_size=0.2, scale_method='standard', feature_list=None):
        """
        Chia data thành train/test và scale features — theo đúng phương pháp luận văn.

        Phương pháp luận văn (paper gốc dùng R):
          - Chia 80/20 với set.seed(123) trong R
          - Stratified sampling (caret package)
          - Scale() cho KNN và SVM

        Cách triển khai ở đây (Python):
          - 80/20 với random_state=123 (tương đương set.seed)
          - stratify=y: đảm bảo tỉ lệ bệnh/không bệnh giống nhau ở train và test
          - StandardScaler thay cho scale() của R

        Ví dụ với 918 mẫu, tỉ lệ bệnh 55%:
          → Train: 734 mẫu (55% = 404 bệnh, 45% = 330 không bệnh)
          → Test:  184 mẫu (55% = 101 bệnh, 45% =  83 không bệnh)

        Tại sao scale?
          StandardScaler: x_scaled = (x - mean) / std
          → Tất cả features có mean=0, std=1
          → KNN: khoảng cách Euclidean không bị ảnh hưởng bởi đơn vị đo
            (ví dụ: tuổi 0-100 và cholesterol 100-600 được đưa về cùng scale)
          → SVM: tìm hyperplane tối ưu chính xác hơn khi features cùng scale
          → Tree-based models (RF, XGBoost): không cần scale, nhưng scale không ảnh hưởng xấu

        QUAN TRỌNG: scaler.fit() CHỈ trên X_train, rồi transform() cả train lẫn test
          → Tránh data leakage: thông tin từ test set không được "rò rỉ" vào scaler
        """
        # Chọn features: ưu tiên feature_list nếu có (từ ACO), không thì dùng 6 features paper
        if feature_list is not None:
            X = df[feature_list]
        else:
            # 6 features theo paper gốc (dùng cho chế độ Baseline)
            paper_features     = ['sex', 'chest pain type', 'fasting blood sugar',
                                   'resting ecg', 'exercise angina', 'ST slope']
            available_features = [f for f in paper_features if f in df.columns]
            X = df[available_features] if available_features else df.drop(columns=[target_col])

        y = df[target_col]

        self.feature_names = X.columns.tolist()
        self.target_name   = target_col

        # Chia stratified 80/20
        # stratify=y: đảm bảo cả train và test có cùng tỉ lệ class
        # shuffle=True: xáo trộn data trước khi chia (tránh data có thứ tự theo thời gian)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y, shuffle=True
        )

        # Scale features theo method được chọn
        if scale_method == 'standard':
            # StandardScaler: (x - mean) / std → mean=0, std=1
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),   # fit+transform trên train
                columns=X_train.columns, index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),         # chỉ transform trên test (không fit lại)
                columns=X_test.columns, index=X_test.index
            )
        elif scale_method == 'minmax':
            # MinMaxScaler: (x - min) / (max - min) → tất cả values trong [0, 1]
            self.scaler = MinMaxScaler()
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns, index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns, index=X_test.index
            )
        # scale_method == 'none': không scale (phù hợp cho tree-based models thuần túy)

        return X_train, X_test, y_train, y_test


# =============================================================================
# TIỀN XỬ LÝ DỮ LIỆU BỆNH VIỆN VN (Long format → Wide format)
# =============================================================================

import re as _re


def _parse_ketqua(value):
    """
    Chuyển đổi giá trị 'ketqua' (kết quả xét nghiệm) từ text thô → số thực.
    Dữ liệu bệnh viện VN thường có nhiều dạng không đồng nhất.

    8 bước xử lý theo thứ tự ưu tiên:

    Bước 1: Số trực tiếp           "13.5"       → 13.5
    Bước 2: Số âm có khoảng trắng  "- 3.8"      → -3.8
    Bước 3: Số kèm đơn vị          "225h"        → 225.0
    Bước 4: Âm tính định tính      "âm tính"     → 0.0
    Bước 5: Dương tính định tính   "dương tính"  → 1.0
    Bước 6: Vết (trace)             "trace"       → 0.5
    Bước 7: Bán định lượng          "1+"          → 1.0, "2+" → 2.0
    Bước 8: Không xác định          "N/A", "?"    → NaN (loại bỏ)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()

    # Bước 1: Số thực/nguyên trực tiếp
    try:
        return float(s)
    except ValueError:
        pass

    # Bước 2: Số âm có khoảng trắng thừa, ví dụ "- 3.8" → -3.8
    m = _re.match(r'^-\s+([\d.]+)$', s)
    if m:
        try:
            return float('-' + m.group(1))
        except ValueError:
            pass

    # Bước 3: Số kèm đơn vị chữ, ví dụ "225h" → 225.0, "14mmol" → 14.0
    m = _re.match(r'^([\d.]+)[a-zA-Z]+$', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Bước 4: Kết quả âm tính định tính → 0.0 (không có)
    if s.lower() in ('negative', 'âm tính', 'am tinh', 'âm', 'am'):
        return 0.0

    # Bước 5: Kết quả dương tính định tính → 1.0 (có)
    if s.lower() in ('positive', 'dương tính', 'duong tinh', 'dương', 'duong'):
        return 1.0

    # Bước 6: "trace" = vết, lượng rất nhỏ → 0.5 (giữa âm và dương)
    if s.lower() == 'trace':
        return 0.5

    # Bước 7: Bán định lượng kiểu "+", ví dụ "1+" → 1.0, "2+" → 2.0
    m = _re.match(r'^(\d+)\+$', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Bước 8: Tất cả trường hợp còn lại → không xác định → loại bỏ
    return np.nan


def preprocess_vn_data(df_raw):
    """
    Chuyển đổi dữ liệu bệnh viện VN từ LONG FORMAT sang WIDE FORMAT.

    Tại sao cần chuyển đổi?
      Data gốc (long): mỗi dòng = 1 xét nghiệm của 1 bệnh nhân
        mavaovien   | maxn   | ketqua | tuoi | phai
        210929...   | GFR001 | 13.35  | 45   | 0
        210929...   | H06    | 30     | 45   | 0
        210929...   | H23.7  | 14.3   | 45   | 0

      Data sau xử lý (wide): mỗi dòng = 1 bệnh nhân, mỗi xét nghiệm = 1 cột
        target | tuoi | phai | GFR001 | H06 | H23.7 | ...
        0      | 45   | 0    | 13.35  | 30  | 14.3  | ...

    Hỗ trợ 3 định dạng file:
      Format A: có cột 'icd_level_0' + 'tenxn' → target = (icd_level_0 == 'IX')
      Format B: có cột 'is_direct_cardio'       → target = is_direct_cardio (0/1)
      Format C: có cột 'target' sẵn             → dùng trực tiếp

    Các bước xử lý:
      1. Xóa duplicates, bỏ hàng thiếu mavaovien
      2. Parse ketqua → số (8 bước)
      3. Xây dựng target nhị phân theo mavaovien
      4. Pivot maxn → cột (mean nếu 1 bệnh nhân có nhiều lần xét nghiệm cùng loại)
      5. Gộp demographics: tuoi, phai, mach, tam_truong, tam_thu
      6. Lọc bỏ cột quá thưa (>99% NaN)
    """
    df = df_raw.copy()

    # --- Xác định định dạng file ---
    has_direct_target = "is_direct_cardio" in df.columns or "target" in df.columns
    direct_target_col = "target" if "target" in df.columns else "is_direct_cardio"
    has_tenxn         = "tenxn" in df.columns  # có tên xét nghiệm đầy đủ không

    # --- Làm sạch cơ bản ---
    df = df.drop_duplicates()
    df = df.dropna(subset=["mavaovien"])  # bỏ hàng không có mã vào viện

    # Bỏ hàng mà CẢ HAI maxn và ketqua đều thiếu (hàng vô nghĩa)
    both_missing = df["maxn"].isna() & df["ketqua"].isna()
    df = df[~both_missing]

    # Chuyển đổi ketqua → số thực (8-step parsing)
    df["ketqua"] = df["ketqua"].apply(_parse_ketqua)
    df = df.dropna(subset=["ketqua"])  # bỏ hàng không parse được

    # --- Xây dựng nhãn target cho từng lần vào viện ---
    if has_direct_target:
        # Format B/C: target đã là 0/1, lấy giá trị đầu tiên của mỗi lần vào viện
        visit_target = (
            df.groupby("mavaovien")[direct_target_col]
            .first()
            .rename("target")
            .reset_index()
        )
    else:
        # Format A: bệnh tim nếu lần vào viện có ít nhất 1 mã ICD nhóm IX
        # (Chương IX của ICD-10 = Bệnh hệ tuần hoàn)
        df["icd_level_0"] = df["icd_level_0"].fillna("unknown")
        visit_target = (
            df.groupby("mavaovien")["icd_level_0"]
            .apply(lambda x: int((x == "IX").any()))  # 1 nếu có bất kỳ ICD IX nào
            .rename("target")
            .reset_index()
        )

    # --- Pivot: maxn → cột, ketqua → giá trị (mean nếu nhiều lần xét nghiệm) ---
    # Ví dụ: bệnh nhân 210929... có 2 lần xét nghiệm H06 (kết quả 28 và 32)
    # → cột H06 của bệnh nhân này = (28 + 32) / 2 = 30
    pivot = (
        df.groupby(["mavaovien", "maxn"])["ketqua"]
        .mean()
        .unstack(level="maxn")  # mỗi maxn thành 1 cột
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # --- Lấy thông tin demographics: tuoi, phai + các vitals cố định ---
    first_per_visit = df.groupby("mavaovien").first().reset_index()

    # Các vitals cố định (lấy nếu có trong file)
    optional_vitals  = ["tam_truong", "tam_thu", "mach"]
    extra_vital_cols = [c for c in optional_vitals if c in first_per_visit.columns]
    demo_cols        = ["mavaovien", "tuoi", "phai"] + extra_vital_cols
    demo             = first_per_visit[demo_cols].copy()

    # Encode giới tính: Nam=1, Nữ=0 (để model xử lý được)
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

    # --- Gộp tất cả thành 1 bảng ---
    # target (nhãn) + demographics + kết quả xét nghiệm
    result = visit_target.merge(demo,  on="mavaovien", how="left")
    result = result.merge(pivot,       on="mavaovien", how="left")
    result = result.drop(columns=["mavaovien"])  # bỏ mã định danh, không cần cho ML

    # Đảm bảo tất cả cột đều là số (object columns → NaN nếu không parse được)
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # --- Lọc bỏ cột quá thưa ---
    # core_cols được bảo vệ: không lọc dù có nhiều NaN
    # feature_cols (xét nghiệm): lọc nếu toàn NaN hoặc >99% NaN
    core_cols    = ["target", "tuoi", "phai"] + extra_vital_cols
    feature_cols = [c for c in result.columns if c not in core_cols]

    # Bỏ cột toàn NaN (xét nghiệm không có kết quả nào)
    all_nan_feats = [c for c in feature_cols if result[c].isna().all()]
    if all_nan_feats:
        result = result.drop(columns=all_nan_feats)

    # Bỏ cột >99% NaN (quá ít dữ liệu để impute)
    feature_cols  = [c for c in result.columns if c not in core_cols]
    sparse_feats  = [c for c in feature_cols if result[c].isna().mean() > 1]
    if sparse_feats:
        result = result.drop(columns=sparse_feats)

    # --- Tạo mapping maxn → tên xét nghiệm đầy đủ (chỉ Format A) ---
    # Dùng để hiển thị tên đẹp trên giao diện thay vì mã code
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
    """
    Tách X (features) và y (target) từ DataFrame để đưa vào ACO.
    Tất cả cột trừ target_col đều là features tiềm năng cho ACO khám phá.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
