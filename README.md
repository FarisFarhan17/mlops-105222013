# MLFlow Task Report - Faris Farhan (105222013)

## Task 1: Environment Setup
a.
![MLflow UI](assets/start_ss.png)
b.
![MLflow UI](assets/requirements_ss.png)

---

## Task 2: Synthetic Data Results

![MLflow UI](assets/1_1.png)

Proyek ini melibatkan pelatihan dan logging dua model machine learning, yaitu **Random Forest** dan **Logistic Regression**, menggunakan dataset churn. Kita akan menganalisis performa kedua model berdasarkan tiga aspek utama: evaluasi akurasi dan AUC score, parameter model, dan metadata run.

### a. Logistic Regression

![MLflow UI](assets/1_2.png)

### b. Random Forest

![MLflow UI](assets/1_3.png)

### 1. **Evaluasi Accuracy dan AUC Score**

- **Accuracy**:
  - **Random Forest**: 0.69
  - **Logistic Regression**: 0.69
  - **Interpretasi**: Kedua model memiliki akurasi yang sama sebesar 69%. Ini menunjukkan bahwa dari seluruh prediksi yang dibuat oleh model, 69% di antaranya adalah benar. Namun, akurasi ini tidak selalu menjadi metrik terbaik untuk mengevaluasi performa model, terutama dalam dataset yang mungkin tidak seimbang.

- **AUC (Area Under the Curve)**:
  - **Random Forest**: 0.7388
  - **Logistic Regression**: 0.7559
  - **Interpretasi**: Nilai AUC Logistic Regression lebih tinggi dibandingkan Random Forest. AUC mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. Dengan AUC yang lebih tinggi, Logistic Regression lebih baik dalam mengidentifikasi kelas churn yang benar dibandingkan Random Forest, meskipun selisihnya tidak terlalu besar.

### 2. **Melihat Parameter Model**

- **Random Forest**:
  - Parameter yang digunakan adalah:
    - `n_estimators`: 100 (jumlah pohon dalam hutan)
    - `max_depth`: 5 (kedalaman maksimum setiap pohon)
    - `random_state`: 42 (untuk memastikan hasil yang dapat direproduksi)
  - **Interpretasi**: Parameter `max_depth` yang dibatasi hingga 5 bertujuan untuk mencegah overfitting dengan mengurangi kompleksitas pohon. `n_estimators` yang sebesar 100 menunjukkan bahwa model ini menggunakan cukup banyak pohon untuk memberikan prediksi yang stabil.

- **Logistic Regression**:
  - Tidak ada parameter yang dilog di MLflow. Model ini menggunakan pengaturan default:
    - Regulasi `L2` untuk mencegah overfitting.
    - Solver `lbfgs` yang cocok untuk dataset dengan jumlah fitur yang relatif kecil hingga sedang.
  - **Interpretasi**: Logistic Regression adalah model linier sederhana yang biasanya tidak memerlukan banyak tuning parameter. Ini membuatnya lebih mudah untuk diimplementasikan dan diinterpretasikan.

### 3. **Memeriksa Metadata Run**

- **Durasi Pelatihan**:
  - **Random Forest**: 10.7 detik
  - **Logistic Regression**: 5.6 detik
  - **Interpretasi**: Logistic Regression memerlukan waktu pelatihan yang lebih singkat dibandingkan Random Forest. Hal ini disebabkan oleh sifat modelnya yang lebih sederhana dan kurang kompleks dibandingkan dengan Random Forest yang melibatkan banyak pohon keputusan.

Metadata lain yang dicatat selama pelatihan mencakup informasi tentang durasi pelatihan, jumlah pohon yang digunakan dalam Random Forest, kedalaman maksimum pohon, serta fitur-fitur yang dilibatkan dalam pelatihan. Data ini penting untuk analisis lanjutan, seperti evaluasi performa model di masa mendatang, tuning parameter, dan pemantauan drift data. Logistic Regression, meskipun tidak mencatat parameter eksplisit di MLflow, tetap menyediakan metrik performa yang relevan dan file JSON berisi nama fitur, yang berguna untuk mendeteksi perubahan pada data input di masa depan.

### Kesimpulan

Logistic Regression menunjukkan keunggulan dalam kecepatan pelatihan dan AUC yang lebih tinggi, membuatnya lebih efisien dan cukup efektif untuk dataset churn ini. Sementara Random Forest memerlukan lebih banyak waktu pelatihan karena kompleksitasnya, model ini menawarkan fleksibilitas yang lebih tinggi dalam mengatur parameter untuk mengatasi data yang lebih kompleks.

**Analisis Singkat Model Terbaik**: Logistic Regression memiliki performa yang sedikit lebih baik dalam hal AUC, dengan skor 0.7559 dibandingkan dengan 0.7388 dari Random Forest. Ini menunjukkan bahwa Logistic Regression lebih andal dalam membedakan antara pelanggan yang akan churn dan yang tidak. Meskipun kedua model memiliki akurasi yang sama sebesar 69%, perbedaan AUC ini menunjukkan bahwa Logistic Regression lebih efektif dalam situasi di mana probabilitas hasil lebih penting daripada sekadar prediksi klasifikasi yang tepat. Selain itu, Logistic Regression juga memerlukan waktu pelatihan yang jauh lebih singkat (5.6 detik dibandingkan 10.7 detik untuk Random Forest), membuatnya lebih praktis untuk digunakan dalam aplikasi dunia nyata yang memerlukan pembaruan model secara berkala atau saat data baru tersedia dengan cepat.

**Polanya yang Menarik**: Salah satu pola menarik yang muncul dari analisis ini adalah bahwa meskipun kedua model memiliki akurasi yang sama, Logistic Regression memiliki AUC yang lebih tinggi. Ini menunjukkan bahwa Logistic Regression lebih baik dalam memberikan probabilitas yang lebih akurat untuk setiap prediksi, yang sangat berguna dalam pengambilan keputusan berbasis risiko, seperti menentukan pelanggan mana yang perlu ditindaklanjuti untuk mengurangi churn. Sebaliknya, Random Forest, dengan akurasi yang sama namun AUC lebih rendah, mungkin menghasilkan prediksi probabilitas yang kurang akurat, meskipun masih cukup baik untuk beberapa kasus penggunaan. Pola ini menggarisbawahi pentingnya tidak hanya melihat akurasi, tetapi juga mempertimbangkan metrik lainnya seperti AUC untuk memahami performa model secara lebih menyeluruh.

---

## Task 2: Dataset Documentation

### Titanic Dataset Logging with MLflow

#### Deskripsi Dataset

Dataset yang digunakan dalam proyek ini adalah dataset Titanic yang digunakan untuk memprediksi apakah seorang penumpang selamat atau tidak dalam peristiwa tenggelamnya kapal Titanic. Dataset ini sering digunakan dalam tugas pembelajaran mesin untuk klasifikasi biner.

Dataset ini berisi informasi tentang penumpang Titanic, yang terdiri dari beberapa fitur yang dapat mempengaruhi kemungkinan selamatnya penumpang. Fitur-fitur yang tersedia antara lain:

- **Pclass**: Kelas penumpang (1, 2, atau 3).
- **Sex**: Jenis kelamin penumpang (laki-laki atau perempuan).
- **Age**: Usia penumpang (dalam tahun).
- **SibSp**: Jumlah saudara kandung atau pasangan yang ikut.
- **Parch**: Jumlah orang tua atau anak yang ikut.
- **Fare**: Tarif yang dibayar penumpang.
- **Embarked**: Pelabuhan keberangkatan penumpang (C = Cherbourg, Q = Queenstown, S = Southampton).
  
Dataset ini memiliki total 891 baris data dan 12 kolom, yang berisi informasi tentang penumpang dan apakah mereka selamat atau tidak. Beberapa kolom seperti `Age` dan `Cabin` memiliki nilai yang hilang, sehingga penanganan data yang hilang diperlukan dalam analisis lebih lanjut.

#### Screenshot 

1.
![MLflow UI](assets/2_1.png)

2.
![MLflow UI](assets/2_2.png)

3.
![MLflow UI](assets/2_3.png)

---

## Task 3: Custom Dataset Results

1.
![MLflow UI](assets/3_1.png)

2.
![MLflow UI](assets/3_2.png)

3.
![MLflow UI](assets/3_3.png)

4. 
![MLflow UI](assets/3_4.png)

### Hasil Performa Model

- **Logistic Regression**:
  - **Akurasi**: 80,45%
  - **AUC**: 0,8768
  - **Durasi Pelatihan**: 20,1 detik

- **Random Forest**:
  - **Akurasi**: 81,01%
  - **AUC**: 0,8955
  - **Durasi Pelatihan**: 8,3 detik

- **Neural Network**:
  - **Akurasi**: 80,45%
  - **AUC**: 0,8506
  - **Durasi Pelatihan**: 6,5 detik

### Hasil Perbandingan Model dan Analisis Kinerja

#### Akurasi
- **Random Forest** menampilkan akurasi tertinggi dengan **81,0%**, menunjukkan sedikit keunggulan dibandingkan **Logistic Regression** dan **Neural Network**, yang memiliki akurasi **80,4%**.
- Akurasi yang sedikit lebih tinggi dari Random Forest menunjukkan bahwa model ini lebih dapat diandalkan untuk prediksi yang benar dibandingkan dua model lainnya.

#### AUC (Area Under the Curve)
- **Random Forest** unggul dalam hal AUC, dengan skor **0,8955**, menandakan kemampuan yang lebih baik dalam membedakan antara kelas yang bertahan hidup dan tidak bertahan hidup.
- **Logistic Regression** memiliki AUC **0,8768**, yang masih cukup baik, meskipun tidak sebaik Random Forest.
- **Neural Network** mencatatkan AUC terendah, **0,8506**, yang menunjukkan performa yang baik tetapi kurang efektif dalam membedakan kelas dibandingkan dua model lainnya.

#### Durasi Pelatihan
- **Neural Network** membutuhkan waktu pelatihan tercepat, yaitu **6,5 detik**, diikuti oleh **Random Forest** dengan **8,3 detik**.
- **Logistic Regression** memerlukan waktu paling lama, **20,1 detik**, karena proses iteratifnya yang lebih kompleks untuk mencapai konvergensi.

### Kesimpulan
**Random Forest** menunjukkan performa terbaik dalam hal akurasi dan AUC, membuatnya menjadi pilihan yang paling optimal untuk dataset Titanic ini. Meskipun **Logistic Regression** dan **Neural Network** memberikan hasil yang kompetitif, durasi pelatihan yang lebih lama pada Logistic Regression dan AUC yang lebih rendah pada Neural Network menjadikan mereka pilihan yang kurang optimal dalam konteks ini.

---

## Task 4: Drift Detection

### Screenshot

1.
![MLflow UI](assets/done.png)

2.
![MLflow UI](assets/4_1.png)

3.
![MLflow UI](assets/4_2.png)

4. 
![MLflow UI](assets/4_3.1.png)
![MLflow UI](assets/4_3.2.png)
![MLflow UI](assets/4_3.3.png)
![MLflow UI](assets/4_4.png)

Pada eksperimen ini, dilakukan deteksi drift pada tiga fitur dari dataset Titanic, yaitu `Age`, `Fare`, dan `Pclass`. Drift dapat mengindikasikan perubahan signifikan pada distribusi data antara dataset pelatihan dan dataset pengujian yang dapat mempengaruhi performa model. Berdasarkan hasil perhitungan **Population Stability Index (PSI)**, berikut adalah analisis deteksi drift untuk masing-masing fitur:

### 1. Hasil Deteksi Drift Fitur

#### **a. Age (Usia)**
- **PSI**: 0.0417
- **Status Drift**: **Tidak ada drift**

Fitur `Age` menunjukkan nilai PSI sebesar **0.0417**, yang jauh di bawah batas nilai PSI 0.1. Nilai PSI yang rendah ini mengindikasikan bahwa distribusi data `Age` pada dataset pelatihan dan pengujian sangat mirip dan tidak mengalami perubahan signifikan. Hal ini menunjukkan bahwa fitur `Age` tetap stabil dan tidak mengalami drift antara data pelatihan dan data uji, sehingga model yang menggunakan fitur ini kemungkinan besar akan tetap stabil di masa depan.

#### **b. Fare (Tarif)**
- **PSI**: 0.0217
- **Status Drift**: **Tidak ada drift**

Fitur `Fare` juga menunjukkan nilai PSI yang sangat rendah, yaitu **0.0217**. Ini menandakan bahwa distribusi nilai `Fare` pada dataset pelatihan dan dataset pengujian tetap konsisten dan tidak menunjukkan adanya pergeseran signifikan. Dengan kata lain, tidak ada perubahan distribusi yang cukup besar yang dapat mempengaruhi performa model dalam menggunakan fitur `Fare`. Model dapat terus mengandalkan fitur ini dengan keyakinan bahwa distribusinya tetap stabil.

#### **c. Pclass (Kelas Penumpang)**
- **PSI**: 0.0036
- **Status Drift**: **Tidak ada drift**

Fitur `Pclass` yang menggambarkan kelas penumpang (1, 2, atau 3) memiliki nilai PSI terendah, yaitu **0.0036**. Nilai ini jauh di bawah batas 0.1 dan menunjukkan bahwa distribusi data pada fitur ini juga sangat stabil antara dataset pelatihan dan pengujian. Karena `Pclass` adalah fitur kategorikal dengan sedikit nilai unik, perbedaan distribusi antara pelatihan dan pengujian

