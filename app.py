import streamlit as st
import pandas as pd
import pickle

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Kelayakan Klaim BPJS",
    page_icon="🏥",
    layout="wide"
)

# 2. Fungsi untuk Memuat Model dan Komponen dari Pickle
@st.cache_resource
def load_model():
    with open('model_bpjs.pkl', 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data['model'], pickle_data['feature_names'], pickle_data['label_encoder']

try:
    model, feature_names, le = load_model()
except FileNotFoundError:
    st.error("⚠️ File 'model_bpjs.pkl' tidak ditemukan. Pastikan file pkl berada dalam satu folder yang sama dengan app.py.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memuat file model: {e}")
    st.stop()

# 3. Judul dan Deskripsi Aplikasi
st.title("🏥 Sistem Prediksi Kelayakan Klaim BPJS")
st.write(
    "Aplikasi ini menggunakan model **Decision Tree** untuk memprediksi status kelayakan klaim "
    "(**Layak** atau **Pending**) berdasarkan variabel perawatan dan biaya."
)

# 4. Tab Antarmuka (Input Tunggal & Batch)
tab1, tab2 = st.tabs(["📝 Input Tunggal", "📁 Unggah File (Batch)"])

# -------------------------------------------------------------------
# TAB 1: PREDIKSI TUNGGAL
# -------------------------------------------------------------------
with tab1:
    st.subheader("Masukkan Informasi Detail Klaim")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kelas_rawat = st.selectbox(
            "Kelas Rawat", 
            options=[1, 2, 3], 
            index=2,
            help="1: Kelas 1, 2: Kelas 2, 3: Kelas 3"
        )
        los = st.number_input(
            "LOS (Length of Stay / Lama Rawat Dalam Hari)", 
            min_value=1, 
            max_value=100, 
            value=3
        )
        total_tarif = st.number_input(
            "Total Tarif / Tarif INACBG (Rp)", 
            min_value=0, 
            value=3500000, 
            step=50000
        )
        
    with col2:
        tarif_rs = st.number_input(
            "Tarif Rumah Sakit (Rp)", 
            min_value=0, 
            value=3000000, 
            step=50000
        )
        
        # Perhitungan nilai Laba otomatis (TOTAL_TARIF - TARIF_RS)
        laba_default = int(total_tarif - tarif_rs)
        laba = st.number_input(
            "Laba / Selisih Biaya (Rp)", 
            value=laba_default, 
            step=50000,
            help="Nilai positif berarti untung, nilai negatif berarti rugi."
        )

    if st.button("🔍 Cek Status Klaim", type="primary"):
        # Membuat dictionary input
        input_dict = {
            'KELAS_RAWAT': kelas_rawat,
            'LOS': los,
            'TOTAL_TARIF': total_tarif,
            'TARIF_RS': tarif_rs,
            'LABA': laba
        }
        
        # Konversi ke DataFrame dan selaraskan kolom sesuai feature_names (mencegah KeyError)
        df_input = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        
        # Prediksi
        pred_index = model.predict(df_input)[0]
        pred_label = le.inverse_transform([pred_index])[0]
        
        # Tampilkan Hasil
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
    st.subheader("Unggah Berkas Data Klaim (CSV / Excel)")
    uploaded_file = st.file_uploader("Pilih berkas", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
                
            st.write("Preview Data Asli:", df_batch.head())
            
            # Hitung kolom LABA secara otomatis jika belum ada di file
            if 'LABA' not in df_batch.columns and 'TOTAL_TARIF' in df_batch.columns and 'TARIF_RS' in df_batch.columns:
                df_batch['LABA'] = df_batch['TOTAL_TARIF'] - df_batch['TARIF_RS']
                st.info("💡 Kolom 'LABA' dihitung otomatis dari (TOTAL_TARIF - TARIF_RS).")

            if st.button("🚀 Prediksi Seluruh Data"):
                # Selaraskan kolom data batch dengan daftar fitur model secara aman
                X_batch = df_batch.reindex(columns=feature_names, fill_value=0)
                
                # Prediksi
                preds = model.predict(X_batch)
                df_batch['PREDIKSI_KETERANGAN'] = le.inverse_transform(preds)
                
                st.success("🎉 Proses prediksi selesai!")
                st.dataframe(df_batch)
                
                # Unduh Hasil dalam bentuk CSV
                csv_data = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=csv_data,
                    file_name="hasil_prediksi_bpjs.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses berkas: {e}")