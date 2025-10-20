import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import os

# --- 1. Load Model (Penting: Pastikan file 'knn_pinjam_mod.pkl' ada di direktori yang sama) ---
try:
    with open('knn_pinjam_mod.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("ERROR: File model 'knn_pinjam_mod.pkl' tidak ditemukan. Pastikan file model sudah tersedia.")
    knn_model = None

st.title('K-Nearest Neighbors (KNN) Model for Loan Approval Prediction')

# --- 2. Input Data dari Sidebar ---
st.sidebar.header('Input Data Calon Peminjam')

usia = st.sidebar.number_input('Usia', min_value=18, max_value=70, value=50)
pendapatan = st.sidebar.number_input('Pendapatan (Juta)', min_value=10, max_value=200, value=50)

# Status Perkawinan (Pemetaan: Belum Menikah=0, Menikah/Cerai=1)
status_perkawinan_input = st.sidebar.selectbox('Status Perkawinan', options=['Belum Menikah', 'Menikah', 'Cerai'])
status_perkawinan = 0 # Default
if status_perkawinan_input == 'Belum Menikah':
    status_perkawinan = 0
else:
    # 'Menikah' atau 'Cerai' akan bernilai 1
    status_perkawinan = 1

jumlah_pinjaman = st.sidebar.number_input('Jumlah Pinjaman (Juta)', min_value=10, max_value=500, value=100)
durasi_pinjaman = st.sidebar.number_input('Durasi Pinjaman (bulan)', min_value=1, max_value=30, value=10)

# Status Pekerjaan (Pemetaan harus lengkap)
status_pekerjaan_input = st.sidebar.selectbox('Status Pekerjaan', 
                                              options=['Karyawan Kontrak', 'Karyawan Tetap', 'Wiraswasta', 'Pensiunan', 'Wirausaha'])

status_pekerjaan = -1 # Nilai default, untuk jaga-jaga

if status_pekerjaan_input == 'Karyawan Kontrak':
    status_pekerjaan = 0
elif status_pekerjaan_input == 'Karyawan Tetap':
    status_pekerjaan = 1
elif status_pekerjaan_input == 'Wiraswasta':
    status_pekerjaan = 2
elif status_pekerjaan_input == 'Pensiunan':
    status_pekerjaan = 3
elif status_pekerjaan_input == 'Wirausaha':
    # BUG FIX: Menambahkan pemetaan untuk 'Wirausaha'
    status_pekerjaan = 4
else:
    # Jaga-jaga jika ada input yang tidak terdefinisi
    st.warning("Peringatan: Status Pekerjaan tidak terdefinisi. Menggunakan nilai -1.")


# --- 3. Prediksi ---
# PENTING: Urutan fitur (kolom) pada new_data HARUS SAMA dengan urutan fitur saat model dilatih.
# Urutan yang digunakan di sini: [usia, pendapatan, status_perkawinan, jumlah_pinjaman, durasi_pinjaman, status_pekerjaan]
new_data = np.array([[float(usia), float(pendapatan), float(status_perkawinan), 
                      float(jumlah_pinjaman), float(durasi_pinjaman), float(status_pekerjaan)]])

if st.sidebar.button('Predict') and knn_model is not None:
    try:
        prediction = knn_model.predict(new_data)
        
        # Asumsi: 1 = Layak (Disetujui), 0 = Tidak Layak (Ditolak)
        result = 'Layak Disetujui' if prediction[0] == 1 else 'Tidak Layak Ditolak'
        
        st.success(f'Hasil Prediksi Kredit: **{result}**')
        
        # --- 4. Visualisasi (Menggunakan Data Random untuk Ilustrasi) ---
        st.subheader('Ilustrasi Data Kredit (Data Acak)')
        
        # Pastikan kolom Lulus_Kredit bertipe string agar Plotly memperlakukannya sebagai kategori warna
        df = pd.DataFrame({
            'Usia' : np.random.randint(18, 70, size=60),
            'Pendapatan' : np.random.randint(10, 200, size=60),
            'Jumlah_Pinjaman' : np.random.randint(10, 500, size=60),
            'Lulus_Kredit' : np.random.choice(['Layak (1)', 'Tidak Layak (0)'], size=60)
        })
        
        # Plot 3D Scatter
        fig = px.scatter_3d(df, x='Usia', y='Pendapatan', z='Jumlah_Pinjaman',
                          color='Lulus_Kredit', 
                          title='Distribusi Data Kredit Berdasarkan Usia, Pendapatan, dan Jumlah Pinjaman')
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi atau menampilkan plot: {e}")
        st.warning("Kemungkinan terbesar: Jumlah atau urutan fitur (kolom) pada `new_data` tidak sesuai dengan fitur yang digunakan saat melatih model `knn_pinjam_mod.pkl`.")
