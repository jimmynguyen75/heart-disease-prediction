import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dự đoán Bệnh Tim", page_icon="🫀", layout="centered")

st.markdown("## 🫀 Dự đoán Nguy Cơ Bệnh Tim")
st.write("Nhập thông tin bệnh nhân và nhấn **Dự đoán** để xem kết quả.")
st.markdown("---")

if 'trainer' not in st.session_state or not st.session_state.get('results'):
    st.warning("⚠️ Chưa có mô hình được huấn luyện.")
    st.info("Vui lòng quay lại trang chính và chạy Training Pipeline trước.")
    if st.button("← Quay lại"):
        st.switch_page("app.py")
else:
    trainer      = st.session_state['trainer']
    df_stored    = st.session_state.get('df', None)
    feature_cols = st.session_state.get('trained_features', [])
    maxn_to_tenxn = st.session_state.get('maxn_to_tenxn', {})

    if df_stored is None or not feature_cols:
        st.error("Không tìm thấy dữ liệu. Vui lòng train lại.")
    else:
        input_values = {}
        cols = st.columns(3)

        for idx, feature in enumerate(feature_cols):
            col_idx = idx % 3
            label = f"{maxn_to_tenxn[feature]} ({feature})" if maxn_to_tenxn and feature in maxn_to_tenxn else feature
            with cols[col_idx]:
                min_val     = float(df_stored[feature].min())
                max_val     = float(df_stored[feature].max())
                mean_val    = float(df_stored[feature].mean())
                unique_vals = df_stored[feature].nunique()
                if unique_vals <= 5:
                    options = sorted(df_stored[feature].unique().tolist())
                    input_values[feature] = st.selectbox(label, options=options, key=f"uv_{feature}")
                else:
                    is_integer = df_stored[feature].dropna().apply(lambda x: x == int(x)).all()
                    if is_integer:
                        input_values[feature] = st.number_input(label, min_value=int(min_val), max_value=int(max_val * 1.5), step=1, value=int(mean_val), key=f"uv_{feature}")
                    else:
                        input_values[feature] = st.number_input(label, min_value=min_val, max_value=max_val * 1.5, step=(max_val - min_val) / 100, value=mean_val, key=f"uv_{feature}")

        st.markdown("---")
        col_btn, col_back = st.columns([2, 1])
        with col_btn:
            predict_clicked = st.button("🔮 Dự đoán", type="primary", use_container_width=True)
        with col_back:
            if st.button("← Quay lại", use_container_width=True):
                st.switch_page("app.py")

        if predict_clicked:
            input_df = pd.DataFrame([input_values])
            if 'preprocessor' in st.session_state:
                prep = st.session_state['preprocessor']
                if hasattr(prep, 'scaler') and prep.scaler is not None:
                    input_scaled = prep.scaler.transform(input_df)
                    input_df = pd.DataFrame(input_scaled, columns=feature_cols)

            vote_positive = 0
            vote_negative = 0
            for name, model in trainer.trained_models.items():
                pred = model.predict(input_df)[0]
                if pred == 1:
                    vote_positive += 1
                else:
                    vote_negative += 1

            total    = vote_positive + vote_negative
            risk_pct = vote_positive / total * 100

            st.markdown("---")
            st.markdown("### Kết quả dự đoán")
            if risk_pct >= 50:
                st.error(f"⚠️ **Nguy cơ CAO** — {vote_positive}/{total} mô hình dự đoán **có bệnh tim** ({risk_pct:.0f}%)")
            else:
                st.success(f"✅ **Nguy cơ THẤP** — {vote_negative}/{total} mô hình dự đoán **không có bệnh tim** ({100 - risk_pct:.0f}%)")
            st.progress(int(risk_pct))
