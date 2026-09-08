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
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Judul dan Deskripsi Aplikasi
st.title("🏥 Sistem Prediksi Kelayakan Klaim BPJS")
st.write("Aplikasi ini memprediksi status klaim (**Layak** atau **Pending**) berdasarkan variabel numerik biaya dan lama rawat.")

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
        los = st.number_input("LOS (Length of Stay / Hari Rawat)", min_value=1, max_value=60, value=3)
        total_tarif = st.number_input("Total Tarif / Tarif INACBG (Rp)", min_value=0, value=3500000, step=50000)
        
    with col2:
        tarif_rs = st.number_input("Tarif RS (Rp)", min_value=0, value=3000000, step=50000)
        
        # Perhitungan estimasi laba otomatis (Total Tarif - Tarif RS)
        laba_default = int(total_tarif - tarif_rs)
        laba = st.number_input("Laba / Selisih Tarif (Rp)", value=laba_default, step=50000)

    if st.button("🔍 Cek Status Klaim", type="primary"):
        # Format input sesuai persis dengan feature_names dari model
        df_input = pd.DataFrame([{
            'KELAS_RAWAT': kelas_rawat,
            'LOS': los,
            'TOTAL_TARIF': total_tarif,
            'TARIF_RS': tarif_rs,
            'LABA': laba
        }])[feature_names]
        
        # Prediksi Model
        pred_val = model.predict(df_input)[0]
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
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
                
            st.write("Preview Data Asli:", df_batch.head())
            
            # Cek jika kolom LABA belum ada, hitung otomatis dari TOTAL_TARIF - TARIF_RS
            if 'LABA' not in df_batch.columns and 'TOTAL_TARIF' in df_batch.columns and 'TARIF_RS' in df_batch.columns:
                df_batch['LABA'] = df_batch['TOTAL_TARIF'] - df_batch['TARIF_RS']
                st.info("💡 Kolom 'LABA' dihitung otomatis dari (TOTAL_TARIF - TARIF_RS).")

            # Validasi ketersediaan kolom
            missing_cols = [col for col in feature_names if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ File yang diunggah kekurangan kolom wajib berikut: `{missing_cols}`")
            else:
                if st.button("🚀 Prediksi Seluruh Data"):
                    # Ambil hanya kolom yang dibutuhkan model
                    X_batch = df_batch[feature_names]
                    
                    # Prediksi
                    preds = model.predict(X_batch)
                    df_batch['PREDIKSI_KETERANGAN'] = le.inverse_transform(preds)
                    
                    st.success("Proses prediksi selesai!")
                    st.dataframe(df_batch)
                    
                    # Tombol Unduh Hasil
                    output_csv = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi (CSV)",
                        data=output_csv,
                        file_name="hasil_prediksi_bpjs.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")