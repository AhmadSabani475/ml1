import pickle
import streamlit as st

# aplikasi streamlit
st.title("Prediksi Kalori")
# Form Input
st.header("Masukkan Data")
umur = st.number_input("Umur", min_value=25, max_value=55 )
bb = st.number_input('Berat Badan (kg)', min_value=60, max_value=95)
tb = st.number_input('TInggi Badan (Cm)', min_value=155, max_value=180  )
olahraga = st.number_input('Olahraga (menit)', min_value=20, max_value=90  )
# Tombol Prediksi
if st.button("Prediksi"):
    # Load model
    loaded_model = pickle.load(open('regression_model.pkl', 'rb'))
    #    melakukan prediksi
    input_data = [[umur, bb, tb, olahraga]]
    prediction = loaded_model.predict(input_data)

#    Menampilkan hasil prediksi
    st.header("Hasil Prediksi")
    st.write(f"Kalori yang dibutuhkan per hari adalah : {prediction[0]:.2f} kalori")
