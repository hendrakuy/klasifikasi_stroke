import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import zscore
import joblib

# Fungsi untuk load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('stroke_dataset.csv')
    return data

# Fungsi untuk analisis data
def analisis_data(data):
    # Menampilkan 5 data awal
    st.write("### Tampilan Data Awal")
    st.write("Berikut adalah tampilan data awal dari dataset stroke yang digunakan.")
    st.dataframe(data)

    # Menampilkan statistik deskriptif
    st.write("### Statistik Deskriptif")
    st.write("Statistik deskriptif digunakan untuk merangkum dan menyajikan data secara ringkas agar pola, distribusi, dan karakteristik utama data lebih mudah dipahami.")
    st.write(data.describe())

    # Menampilkan tipe data
    st.write("### Informasi Dataset")
    buffer = pd.DataFrame(data.dtypes, columns=['Data Type'])
    st.dataframe(buffer)

    # Menampilkan distribusi target (kolom stroke)
    st.write("### Distribusi Target Stroke")
    stroke_counts = data['stroke'].value_counts()
    stroke_distribution = data['stroke'].value_counts(normalize=True) * 100

    # Plot distribusi target
    fig, ax = plt.subplots(figsize=(8, 5))
    stroke_counts.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)
    plt.title('Presentasi Stroke', fontsize=18, fontweight='bold')
    plt.xlabel('Stroke Status', fontsize=14)
    plt.ylabel('Jumlah', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Tidak Stroke', 'Stroke'], rotation=0)
    st.pyplot(fig)

    # Menampilkan jumlah dan presentase target stroke
    stroke_summary = pd.DataFrame({
        'Count': stroke_counts,
        'Percentage (%)': stroke_distribution
    })
    st.write("Distribusi Target:")
    st.table(stroke_summary)

    st.write("### Informasi Fitur dan Target")
    st.write("Dataset ini terdiri dari 10 fitur dan 1 target (label) yang digunakan untuk memprediksi apakah seseorang terkena stroke:")
    st.write("**Fitur:**")
    st.markdown("""
    <ul style="list-style-type: none;">
        <li style="margin-left: -10px; font-weight: bold;">1. ID
            <ul style="padding-left: 15px;">
                <li>Identifikasi unik untuk setiap pasien</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">2. Gender
            <ul style="padding-left: 15px;">
                <li>Female: Pasien berjenis kelamin perempuan</li>
                <li>Male: Pasien berjenis kelamin laki-laki</li>
                <li>Other: Jenis kelamin lainnya</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">3. Age
            <ul style="padding-left: 15px;">
                <li>Usia pasien dalam tahun</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">4. Hypertension
            <ul style="padding-left: 15px;">
                <li>No: 0 (Pasien tidak memiliki hipertensi)</li>
                <li>Yes: 1 (Pasien memiliki hipertensi)</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">5. Heart Disease
            <ul style="padding-left: 15px;">
                <li>No: 0 (Pasien tidak memiliki penyakit jantung)</li>
                <li>Yes: 1 (Pasien memiliki penyakit jantung)</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">6. Ever Married
            <ul style="padding-left: 15px;">
                <li>No: Pasien belum pernah menikah</li>
                <li>Yes: Pasien pernah menikah</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">7. Work Type
            <ul style="padding-left: 15px;">
                <li>Govt_job: Pasien bekerja sebagai pegawai pemerintah</li>
                <li>Never_worked: Pasien belum pernah bekerja</li>
                <li>Private: Pasien bekerja di sektor swasta</li>
                <li>Self-employed: Pasien bekerja secara mandiri</li>
                <li>Children: Pasien adalah anak-anak yang belum bekerja</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">8. Residence Type
            <ul style="padding-left: 15px;">
                <li>Urban: Pasien tinggal di perkotaan</li>
                <li>Rural: Pasien tinggal di pedesaan</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">9. Smoking Status
            <ul style="padding-left: 15px;">
                <li>Unknown: Informasi merokok tidak tersedia</li>
                <li>Formerly smoked: Pasien adalah mantan perokok</li>
                <li>Never smoked: Pasien tidak pernah merokok</li>
                <li>Smokes: Pasien saat ini merokok</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">10. Avg Glucose Level
            <ul style="padding-left: 15px;">
                <li>Rata-rata level glukosa dalam darah (mg/dL)</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">11. BMI
            <ul style="padding-left: 15px;">
                <li>Indeks Massa Tubuh (berdasarkan berat dan tinggi badan)</li>
            </ul>
        </li>
    </ul>
    <br>
    <b>Target:</b>
    <ul style="list-style-type: none;">
        <li style="margin-left: -10px; font-weight: bold;">Stroke
            <ul style="padding-left: 15px;">
                <li>No: 0 (Pasien tidak terkena stroke)</li>
                <li>Yes: 1 (Pasien terkena stroke)</li>
            </ul>
        </li>
    </ul>
""", unsafe_allow_html=True)

# Fungsi untuk preprocessing data
def preprocessing_data(data):
    # Menangani missing value
    st.write("### 1. Penanganan Missing Value")
    st.write("Langkah pertama dalam preprocessing data adalah menangani missing value. Missing value ditemukan pada kolom BMI. Penanganan missing value dilakukan dengan mengisi nilai rata-rata untuk kolom BMI.")
    st.write("**Sebelum Penanganan Missing Value:**")
    st.write(data.isnull().sum())
    st.dataframe(data.head())

    # Mengisi nilai null di kolom bmi dengan rata-rata
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    st.write("**Setelah Penanganan Missing Value:**")
    st.write(data.isnull().sum())
    st.dataframe(data.head())

    # Menghapus kolom id
    data = data.drop('id', axis=1)
    
    # Menghapus gender 'Other'
    data = data[data['gender'] != 'Other']

    # Kolom numerik yang dianalisis
    st.write("### 2. Penanganan Outlier")
    st.write("Outlier adalah data yang memiliki nilai ekstrem berbeda dari sebagian besar data lainnya, yang dapat disebabkan oleh kesalahan atau variasi wajar. Outlier perlu diidentifikasi dan ditangani untuk meningkatkan akurasi model.")
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']

    # Menghitung jumlah outlier dengan Z-Score
    st.write("**Jumlah Outlier berdasarkan Z-Score (threshold = 3):**")
    z_scores = np.abs(zscore(data[numeric_columns]))
    threshold = 3
    outlier_counts = (z_scores > threshold).sum(axis=0)

    outlier_summary = pd.DataFrame({
        "Fitur": numeric_columns,
        "Jumlah Outlier": outlier_counts
    })
    st.dataframe(outlier_summary)    

    # Visualisasi sebelum penanganan outliers
    st.write("**Visualisasi Sebelum Penanganan Outliers:**")
    fig_before, ax_before = plt.subplots(1, len(numeric_columns), figsize=(15, 5))
    if len(numeric_columns) == 1:
        ax_before = [ax_before]
    for i, col in enumerate(numeric_columns):
        sns.boxplot(y=data[col], ax=ax_before[i], color='skyblue')
        ax_before[i].set_title(f'Boxplot {col} Sebelum Penanganan')
    st.pyplot(fig_before)

    # Menghitung Z-Score untuk kolom numerik
    z_scores = pd.DataFrame(zscore(data[numeric_columns]), columns=numeric_columns)
    threshold = 3
    outliers = (np.abs(z_scores) > threshold)

    # Mengganti outliers dengan nilai rata-rata kolom
    for col in numeric_columns:
        data[col] = data[col].where(~outliers[col], data[col].mean())
    st.write("**Visualisasi Setelah Penanganan Outliers:**")
    fig_after, ax_after = plt.subplots(1, len(numeric_columns), figsize=(15, 5))
    if len(numeric_columns) == 1:
        ax_after = [ax_after]
    for i, col in enumerate(numeric_columns):
        sns.boxplot(y=data[col], ax=ax_after[i], color='lightgreen')
        ax_after[i].set_title(f'Boxplot {col} Setelah Penanganan')
    st.pyplot(fig_after)

    # Sebelum transformasi data
    st.write("### 3. Transformasi Data")
    st.write("Transformasi data adalah proses mengubah data mentah menjadi format yang lebih sesuai atau berguna untuk analisis atau pemodelan. Label Encoder digunakan untuk melakukan transformasi data kategorikal ke bentuk numerik.")
    st.markdown("""
    <ul style="list-style-type: none;">
        <li style="margin-left: -10px; font-weight: bold;">1. Gender
            <ul style="padding-left: 15px;">
                <li>Perempuan: 0</li>
                <li>Laki-laki: 1</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">2. Ever Married
            <ul style="padding-left: 15px;">
                <li>No: 0</li>
                <li>Yes: 1</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">3. Work Type
            <ul style="padding-left: 15px;">
                <li>Govt_job: 0</li>
                <li>Never_worked: 1</li>
                <li>Private: 2</li>
                <li>Self-employed: 3</li>
                <li>Children: 4</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">4. Residence Type
            <ul style="padding-left: 15px;">
                <li>Urban: 1</li>
                <li>Rural: 0</li>
            </ul>
        </li>
        <li style="margin-left: -10px; font-weight: bold;">5. Smoking Status
            <ul style="padding-left: 15px;">
                <li>Unknown: 0</li>
                <li>Formerly smoked: 1</li>
                <li>Never smoked: 2</li>
                <li>Smokes: 3</li>
            </ul>
        </li>
    </ul>
""", unsafe_allow_html=True)
    
    st.write("**Data Sebelum Transformasi:**")
    st.dataframe(data)

    # Transformasi data kategorikal dengan Label Encoder
    encoder = LabelEncoder()
    for col in ['gender', 'smoking_status', 'work_type', 'Residence_type', 'ever_married']:
        data[col] = encoder.fit_transform(data[col])

    st.write("**Data Setelah Transformasi:**")
    st.dataframe(data)

    # Menampilkan distribusi kelas sebelum dan setelah oversampling
    st.write("### 4. Penanganan Imbalance Data (Random Oversampling)")
    st.write("Imbalance data terjadi ketika jumlah kelas dalam dataset tidak seimbang, yang dapat mempengaruhi kinerja model. Teknik seperti oversampling digunakan untuk mengatasinya.")

    # Sebelum Oversampling
    stroke_counts_before = data['stroke'].value_counts()
    stroke_distribution_before = data['stroke'].value_counts(normalize=True) * 100
    stroke_summary = pd.DataFrame({
        'Count': stroke_counts_before,
        'Percentage (%)': stroke_distribution_before
    })
    st.write("**Distribusi Target Sebelum Diseimbangkan:**")
    st.table(stroke_summary)

    # Oversampling dengan RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X = data.drop('stroke', axis=1)
    Y = data['stroke']
    X_resampled, Y_resampled = ros.fit_resample(X, Y)

    # Setelah Oversampling
    stroke_counts_after = Y_resampled.value_counts()
    stroke_distribution_after = Y_resampled.value_counts(normalize=True) * 100
    stroke_summary = pd.DataFrame({
        'Count': stroke_counts_after,
        'Percentage (%)': stroke_distribution_after
    })
    st.write("**Distribusi Target Setelah Diseimbangkan:**")
    st.table(stroke_summary)

    # Menampilkan grafik distribusi kelas
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Sebelum oversampling
    stroke_counts_before.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax[0])
    ax[0].set_title('Distribusi Target Sebelum Diseimbangkan', fontsize=14)
    ax[0].set_xlabel('Stroke Status', fontsize=12)
    ax[0].set_ylabel('Jumlah', fontsize=12)
    ax[0].set_xticklabels(['Tidak Stroke', 'Stroke'], rotation=0)

    # Setelah oversampling
    stroke_counts_after.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax[1])
    ax[1].set_title('Distribusi Target Setelah Diseimbangkan', fontsize=14)
    ax[1].set_xlabel('Stroke Status', fontsize=12)
    ax[1].set_ylabel('Jumlah', fontsize=12)
    ax[1].set_xticklabels(['Tidak Stroke', 'Stroke'], rotation=0)

    st.pyplot(fig)

    # Tampilan Streamlit
    st.write("### 5. Splitting Dataset")
    st.write("Dataset dibagi menjadi dua bagian, yaitu data latih (train) dan data uji (test), dengan rasio 80:20. Data latih berfungsi untuk membangun dan melatih model, sedangkan data uji digunakan untuk mengevaluasi kinerja model.")

    # Membagi dataset menjadi data train dan data test
    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Train:**")
        st.dataframe(X_train)
    with col2:
        st.write("**Data Test:**")
        st.dataframe(X_test)

    with col1:
        st.write("**Distribusi Kelas**")
        st.dataframe(Y_train.value_counts())
    with col2:
        st.write("**Distribusi Kelas**")
        st.dataframe(Y_test.value_counts())

    with col1:
        st.write("**Jumlah data dan fitur**")
        st.write(X_train.shape)
    with col2:
        st.write("**Jumlah data dan fitur**")
        st.write(X_test.shape)

    return data
# Fungsi untuk modelling
def modelling(X_train, X_test, Y_train, Y_test):
    st.write("### Evaluasi Model")
    
    # Daftar model yang akan dievaluasi
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    # Menyimpan hasil evaluasi
    results = []
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        # Melatih model dan membuat prediksi
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        # Menghitung akurasi
        acc = accuracy_score(Y_test, Y_pred)
        results.append({"Model": name, "Accuracy": acc})

        # Menampilkan hasil evaluasi
        st.write(f"**{name}**")
        st.write(f"Accuracy: **{acc:.2f}**")
        
        # Menampilkan confusion matrix sebagai heatmap
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Menampilkan classification report dalam bentuk tabel
        st.write("Classification Report:")
        report = classification_report(Y_test, Y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Simpan model terbaik
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    # Menyimpan model terbaik ke file .pkl
    if best_model:
        joblib.dump(best_model, 'stroke_model.pkl')

    # Menampilkan perbandingan akurasi
    st.write("### Perbandingan Akurasi")
    df_results = pd.DataFrame(results)
    fig, ax = plt.subplots()
    sns.barplot(data=df_results, x="Model", y="Accuracy", ax=ax, palette="viridis")
    ax.set_title("Perbandingan Akurasi Model")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

# Fungsi untuk mengonversi input ke bentuk numerik
def convert_input_to_numeric(input_data, df):
    encoder = LabelEncoder()

    # Kolom-kolom kategorikal yang perlu diubah menjadi numerik
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    for col in categorical_columns:
        # Fit encoder hanya pada kolom-kolom kategorikal dari data yang sudah ada
        encoder.fit(df[col])
        input_data[col] = encoder.transform([input_data[col]])[0]
    
    return input_data

# Fungsi klasifikasi untuk prediksi pengguna
def klasifikasi():
    st.title("Klasifikasi")
    
    # Muat model yang sudah dilatih
    clf = joblib.load('stroke_model.pkl') 
    
    # Muat dataset yang sudah diproses
    df = load_data()
    
    # Masukkan data untuk prediksi
    st.write("Masukkan data untuk prediksi:")
    input_data = {}
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    binary_columns = ['hypertension', 'heart_disease']  # Kolom dengan pilihan 0 atau 1

    # Tangani fitur kategorikal, biner, dan numerik secara terpisah
    for col in df.drop(['stroke', 'id'], axis=1).columns:
        if col in categorical_columns:
            categories = df[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}:", categories, index=categories.index(df[col].mode()[0]))
        elif col in binary_columns:
            input_data[col] = st.selectbox(f"{col}:", [0, 1], index=0)
        else:
            input_data[col] = st.number_input(f"{col}:", value=0.0)

    if st.button("Prediksi"):
        # Mengonversi input menjadi numerik
        input_data = convert_input_to_numeric(input_data, df)
        input_df = pd.DataFrame([input_data])
        prediction = clf.predict(input_df)[0]
        
        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        if prediction == 1:
            st.write("Penderita Stroke")
        else:
            st.write("Tidak Penderita Stroke")

def main():
    st.title("Klasifikasi Penyakit Stroke Menggunakan Metode Decision Tree C45")
    st.sidebar.title("Klasifikasi Stroke dengan Decision Tree C45")
    menu = st.sidebar.radio("Pilih Tahapan:", ["Analisis Data", "Pre Processing", "Modelling", "Klasifikasi"])
    
    # Load dataset
    data = load_data()

    if menu == "Analisis Data":
        st.title("Analisis Data")
        analisis_data(data)

    elif menu == "Pre Processing":
        st.title("Preprocessing Data")
        processed_data = preprocessing_data(data)
        # Menyimpan data hasil preprocessing ke session state untuk digunakan di tahap berikutnya
        st.session_state['data'] = processed_data

    elif menu == "Modelling":
        if 'data' in st.session_state:
            st.title("Modelling")
            data = st.session_state['data']
            X = data.drop('stroke', axis=1)
            Y = data['stroke']
            # Menggunakan teknik oversampling untuk mengatasi data tidak seimbang
            ros = RandomOverSampler(random_state=42)
            X_resampled, Y_resampled = ros.fit_resample(X, Y)
            X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

            modelling(X_train, X_test, Y_train, Y_test)

            # Melatih model decision tree
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, Y_train)

            # Menampilkan struktur decision tree yang dihasilkan
            st.write("### Struktur Decision Tree")
            plt.figure(figsize=(12, 10))
            # Kedalaman pohon = 2
            plot_tree(model, feature_names=X.columns, max_depth=2, filled=True, class_names=['0', '1'])
            st.pyplot(plt)
        else:
            st.warning("Lakukan preprocessing terlebih dahulu")

    elif menu == "Klasifikasi":
        klasifikasi()

if __name__ == "__main__":
    main()
