import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Fungsi untuk menghitung PSI (Population Stability Index)
def calculate_psi(expected, actual, num_bins=10):
    """
    Fungsi untuk menghitung PSI (Population Stability Index)
    :param expected: Distribusi data yang diharapkan (misalnya, data pelatihan)
    :param actual: Distribusi data aktual (misalnya, data yang diuji)
    :param num_bins: Jumlah bin untuk discretization
    :return: Nilai PSI
    """
    # Menggunakan imputasi untuk menangani NaN
    imputer = SimpleImputer(strategy="mean")
    expected = imputer.fit_transform(expected.reshape(-1, 1)).flatten()
    actual = imputer.transform(actual.reshape(-1, 1)).flatten()

    # Discretize kedua distribusi untuk menghitung frekuensi
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    
    # Mendiscretisasi data untuk distribusi yang diharapkan dan aktual
    expected_binned = discretizer.fit_transform(expected.reshape(-1, 1)).astype(int)
    actual_binned = discretizer.transform(actual.reshape(-1, 1)).astype(int)

    # Hitung frekuensi per bin
    expected_freq = np.bincount(expected_binned.flatten(), minlength=num_bins) / len(expected)
    actual_freq = np.bincount(actual_binned.flatten(), minlength=num_bins) / len(actual)

    # Tambahkan epsilon untuk mencegah pembagian dengan nol
    epsilon = 1e-10
    expected_freq = np.maximum(expected_freq, epsilon)
    actual_freq = np.maximum(actual_freq, epsilon)

    # Hitung nilai PSI
    psi_value = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    return psi_value, expected_freq, actual_freq

# Fungsi untuk mendeteksi drift dengan PSI untuk beberapa fitur
def detect_drift(train_dataset_path, test_dataset_path, feature_names):
    """
    Fungsi utama untuk mendeteksi drift dengan menghitung PSI untuk beberapa fitur dan menentukan status drift
    :param train_dataset_path: Path ke dataset training
    :param test_dataset_path: Path ke dataset testing
    :param feature_names: List nama kolom fitur yang digunakan untuk mendeteksi drift
    :return: PSI dan status drift untuk setiap fitur
    """
    # Membaca dataset
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)
    
    drift_results = {}  # Menyimpan hasil PSI dan status drift untuk setiap fitur
    
    for feature_name in feature_names:
        expected_feature = train_data[feature_name].values  # Data pelatihan atau distribusi yang diharapkan
        actual_feature = test_data[feature_name].values  # Data baru (untuk diuji)

        # Menghitung PSI
        psi_value, expected_freq, actual_freq = calculate_psi(expected_feature, actual_feature)
        
        # Tentukan status drift berdasarkan nilai PSI
        if psi_value < 0.1:
            drift_status = 'No drift'
        elif 0.1 <= psi_value < 0.2:
            drift_status = 'Moderate drift'
        else:
            drift_status = 'High drift'

        # Log nilai PSI ke MLflow sebagai metrik
        mlflow.log_metric(f'{feature_name}_PSI_Value', psi_value)
        
        # Log distribusi ke MLflow sebagai artifacts
        mlflow.log_artifact(generate_distribution_comparison_plot(expected_freq, actual_freq, feature_name))

        # Simpan hasil untuk setiap fitur
        drift_results[feature_name] = {'PSI': psi_value, 'Drift Status': drift_status}

    # Log keseluruhan hasil drift ke MLflow
    mlflow.log_dict(drift_results, "overall_drift_analysis.json")
    
    # Tampilkan hasil deteksi drift untuk setiap fitur
    for feature_name, result in drift_results.items():
        print(f"Feature: {feature_name}, PSI: {result['PSI']}, Drift Status: {result['Drift Status']}")
        
    return drift_results

# Fungsi untuk menghasilkan grafik perbandingan distribusi untuk tiap fitur
def generate_distribution_comparison_plot(expected_freq, actual_freq, feature_name):
    """
    Fungsi untuk menghasilkan dan menyimpan grafik perbandingan distribusi untuk tiap fitur
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribusi
    ax.plot(expected_freq, label='Expected Distribution', marker='o')
    ax.plot(actual_freq, label='Actual Distribution', marker='x')
    
    ax.set_title(f'Distribution Comparison for {feature_name}')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Simpan grafik sebagai file PNG
    plot_path = f"{feature_name}_distribution_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)

    return plot_path

# Mulai eksperimen MLflow
if __name__ == "__main__":
    with mlflow.start_run():
        # Path ke dataset
        train_dataset_path = 'data/titanic.csv'  # Gantilah dengan path dataset training
        test_dataset_path = 'data/test.csv'    # Gantilah dengan path dataset testing
        
        # Daftar fitur yang ingin diperiksa untuk drift
        feature_names = ['Age', 'Fare', 'Pclass']  # Gantilah dengan nama kolom fitur yang relevan
        
        # Jalankan deteksi drift
        drift_results = detect_drift(train_dataset_path, test_dataset_path, feature_names)
        
        # Menampilkan hasil
        print(f"Drift detection completed. Results: {drift_results}")
