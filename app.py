import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Fungsi untuk menghitung kategori stres
def kategori_stres(skor):
    if skor <= 3:
        return "Rendah"
    elif skor <= 6:
        return "Sedang"
    else:
        return "Tinggi"

# Fungsi untuk mendapatkan saran berdasarkan kategori
def saran_stres(kategori):
    if kategori == "Rendah":
        return "Pertahankan pola hidup sehat Anda!"
    elif kategori == "Sedang":
        return "Coba tingkatkan jam tidur dan kurangi kafein."
    else:
        return "Perhatikan kesehatan mental, kurangi beban belajar, dan konsultasikan dengan ahli."

# Dataset contoh
data = {
    "Nama": [f"Mahasiswa {i+1}" for i in range(50)],
    "Jam_Tidur": [7,6,5,8,4,6,7,5,6,4,8,6,5,7,4,6,5,8,6,4,
                  7,5,6,4,7,5,8,6,4,6,7,5,6,4,8,6,5,7,6,4,
                  5,7,6,4,8,5,6,4,6,4],
    "Jam_Belajar": [2,3,4,2,6,3,1,5,4,7,3,2,3,4,5,6,2,1,5,3,
                    2,4,1,6,3,3,2,5,4,3,2,6,4,7,3,2,5,1,3,5,
                    4,2,6,3,1,4,5,6,3,6],
    "Kafein": [1,1,2,0,2,1,0,2,1,2,0,1,1,1,2,2,1,0,2,2,
               1,2,1,2,0,1,0,2,2,1,0,2,1,2,0,1,0,1,2,2,
               1,0,2,2,0,1,2,2,1,2],
    "Olahraga": [2,1,0,2,0,1,2,0,1,0,2,1,0,2,0,1,0,2,1,0,
                 2,0,1,0,2,0,2,1,0,1,2,0,1,0,2,1,0,2,1,0,
                 0,2,1,0,2,0,1,0,1,0],
}

# Load additional data from Excel if available
excel_file = "UAS SPK TINGKAT STREs.xlsx"
if os.path.exists(excel_file):
    try:
        df_excel = pd.read_excel(excel_file, skiprows=3, header=0)
        df_excel = df_excel[['nama', 'Jam Tidur', 'Jam Belajar', 'Kafein', 'Olahraga']].rename(columns={'nama': 'Nama', 'Jam Tidur': 'Jam_Tidur', 'Jam Belajar': 'Jam_Belajar'})
        # Append to existing data
        for col in data.keys():
            data[col].extend(df_excel[col].tolist())
    except Exception as e:
        st.warning(f"Gagal memuat data dari Excel: {e}")

df = pd.DataFrame(data)

# Hitung skor stres dan kategori
df["Skor_Stres"] = (10 - df["Jam_Tidur"]) + df["Jam_Belajar"] + (df["Kafein"]*2) - df["Olahraga"]
df["Skor_Stres"] = df["Skor_Stres"].apply(lambda x: max(1, min(10, x)))
df["Kategori_Stres"] = df["Skor_Stres"].apply(kategori_stres)

# Model Decision Tree
X = df[["Jam_Tidur","Jam_Belajar","Kafein","Olahraga"]]
y = df["Kategori_Stres"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Klasifikasi Stres Mahasiswa", page_icon="🧠", layout="wide")



st.title("🧠 Sistem Pakar Klasifikasi Tingkat Stres Mahasiswa")


# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.markdown("""
    **Faktor yang Dipertimbangkan:**
    - Jam Tidur: Ideal 7-8 jam
    - Jam Belajar: Jangan berlebihan
    - Kafein: Kurangi konsumsi berlebih
    - Olahraga: Rutin berolahraga baik untuk kesehatan
    """)

    # Visualisasi distribusi kategori stres
    st.subheader("Distribusi Kategori Stres")
    kategori_counts = df["Kategori_Stres"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(kategori_counts.index, kategori_counts.values, color=['green', 'yellow', 'red'])
    ax.set_ylabel("Jumlah Mahasiswa")
    ax.set_title("Distribusi Tingkat Stres")
    st.pyplot(fig)

# Form input dalam kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Masukkan Data Anda")
    with st.form(key="input_form"):
        nama = st.text_input("Nama Lengkap", placeholder="Masukkan nama Anda")
        jam_tidur = st.slider("Jam Tidur per Hari", 0, 10, 7, help="Ideal: 7-8 jam")
        jam_belajar = st.slider("Jam Belajar per Hari", 0, 10, 4, help="Jangan berlebihan")
        kafein = st.selectbox("Konsumsi Kafein", [0,1,2])
        olahraga = st.selectbox("Aktivitas Olahraga", [0,1,2])
        submit_button = st.form_submit_button(label="🔍 Prediksi Stres")

with col2:
    if submit_button:
        if not nama.strip():
            st.error("Harap masukkan nama Anda!")
        else:
            input_data = pd.DataFrame([[jam_tidur, jam_belajar, kafein, olahraga]],
                                      columns=["Jam_Tidur","Jam_Belajar","Kafein","Olahraga"])
            prediksi = model.predict(input_data)[0]
            skor_prediksi = model.predict_proba(input_data)[0]

            st.subheader("🎯 Hasil Prediksi")
            st.success(f"**Nama:** {nama}")
            if prediksi == "Rendah":
                st.success(f"**Kategori Stres:** Rendah 🟢")
            elif prediksi == "Sedang":
                st.warning(f"**Kategori Stres:** Sedang 🟡")
            else:
                st.error(f"**Kategori Stres:** Tinggi 🔴")

            st.info(f"**Saran:** {saran_stres(prediksi)}")

            # Tampilkan probabilitas
            st.subheader("📊 Probabilitas Prediksi")
            prob_df = pd.DataFrame({
                "Kategori": ["Rendah", "Sedang", "Tinggi"],
                "Probabilitas (%)": [f"{p*100:.1f}%" for p in skor_prediksi]
            })
            st.table(prob_df)
    else:
        st.info("Masukkan data di sebelah kiri dan klik 'Prediksi Stres' untuk melihat hasil.")
