import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi Tampilan Halaman
st.set_page_config(
    page_title="Prediksi Kelayakan Klaim BPJS",
    page_icon="🏥",
    layout="wide"
)

# Load Model dari File Pickle
@st.cache_resource
def load_model():
    with open('model_bpjs.pkl', 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data['model'], pickle_data['feature_names'], pickle_data['label_encoder']

try:
    model, feature_names, le = load_model()
except FileNotFoundError:
    st.error("File 'model_bpjs.pkl' tidak ditemukan. Pastikan file pkl berada di folder yang sama dengan app.py.")
    st.stop()

# Judul dan Deskripsi Aplikasi
st.title("🏥 Sistem Prediksi Kelayakan Klaim BPJS")
st.write("Aplikasi ini memprediksi status klaim (**Layak** atau **Pending**) berdasarkan variabel historis medis dan biaya.")

# Tab Antarmuka
tab1, tab2 = st.tabs(["📝 Input Tunggal", "📁 Unggah File (Batch)"])

# -------------------------------------------------------------------
# TAB 1: PREDIKSI TUNGGAL
# -------------------------------------------------------------------
with tab1:
    st.subheader("Masukkan Detail Klaim")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kelas_rawat = st.selectbox("Kelas Rawat", options=[1, 2, 3], index=2)
        los = st.number_input("LOS (Length of Stay / Hari Rawat)", min_value=1, max_value=30, value=3)
        total_tarif = st.number_input("Total Tarif (Rp)", min_value=0, value=3500000, step=50000)
        tarif_rs = st.number_input("Tarif RS (Rp)", min_value=0, value=3000000, step=50000)
        
    with col2:
        laba = st.number_input("Laba (Rp)", value=total_tarif - tarif_rs, step=50000)
        diaglist = st.text_input("Kode Diagnosis (DIAGLIST)", value="E11.9")
        proclist = st.text_input("Kode Prosedur (PROCLIST)", value="87.44;89.52;90.59")
        inacbg = st.text_input("Kode INACBG", value="E-4-10-I")

    if st.button("🔍 Cek Status Klaim", type="primary"):
        # Format input menjadi DataFrame
        df_input = pd.DataFrame([{
            'KELAS_RAWAT': kelas_rawat,
            'LOS': los,
            'TOTAL_TARIF': total_tarif,
            'TARIF_RS': tarif_rs,
            'LABA': laba,
            'DIAGLIST': str(diaglist),
            'PROCLIST': str(proclist),
            'INACBG': str(inacbg)
        }])
        
        # Preprocessing & Alignment Kolom
        df_encoded = pd.get_dummies(df_input)
        df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
        
        # Prediksi Model
        pred_val = model.predict(df_aligned)[0]
        pred_label = le.inverse_transform([pred_val])[0]
        
        # Tampilkan Hasil Prediksi
        st.divider()
        st.subheader("Hasil Prediksi:")
        if pred_label == "Layak":
            st.success(f"✅ Status Klaim Diprediksi: **{pred_label}**")
        else:
            st.warning(f"⚠️ Status Klaim Diprediksi: **{pred_label}**")

# -------------------------------------------------------------------
# TAB 2: PREDIKSI BANYAK DATA (BATCH)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Unggah File Data Klaim")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_batch = pd.read_csv(uploaded_file)
        else:
            df_batch = pd.read_excel(uploaded_file)
            
        st.write("Preview Data Asli:", df_batch.head())
        
        if st.button("🚀 Prediksi Seluruh Data"):
            X_batch = df_batch.copy()
            if 'KETERANGAN' in X_batch.columns:
                X_batch = X_batch.drop(columns=['KETERANGAN'])
                
            for col in ['DIAGLIST', 'PROCLIST', 'INACBG']:
                if col in X_batch.columns:
                    X_batch[col] = X_batch[col].astype(str)
            
            # Align kolom dengan data training
            X_encoded = pd.get_dummies(X_batch)
            X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0)
            
            # Prediksi
            preds = model.predict(X_aligned)
            df_batch['PREDIKSI_KETERANGAN'] = le.inverse_transform(preds)
            
            st.success("Proses prediksi selesai!")
            st.dataframe(df_batch)
            
            # Tombol Unduh
            output_csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=output_csv,
                file_name="hasil_prediksi_bpjs.csv",
                mime="text/csv"
            )
