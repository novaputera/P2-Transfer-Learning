# 🚀 Laporan Proyek: Klasifikasi Komponen Robot Berbasis Transfer Learning (RET503)

**Mata Kuliah:** RET503 — Computer Vision and Deep Learning

**Institusi:** Politeknik Negeri Batam (Program Studi Teknologi Rekayasa Robotika)

**Fokus Proyek:** Penerapan *Deep Learning* dan *Transfer Learning* untuk pengenalan visual komponen modular robotik (**MG90S Servo** vs **PCA9685 / Step-Down Module**).

## 🛠️ 1. Pengantar & Latar Belakang Proyek

Sistem persepsi yang handal menjadi salah satu pilar utama dalam otomatisasi robot modern. Repositori ini mendokumentasikan eksperimen pembuatan model pengenalan citra cerdas menggunakan pendekatan *Transfer Learning*. Alih-alih melatih model jaringan saraf tiruan dari nol (*from scratch*) yang membutuhkan sumber daya masif dan dataset raksasa, proyek ini memanfaatkan model dasar (*pretrained backbone*) yang telah terlatih pada ribuan kelas objek umum, lalu diadaptasi khusus untuk mengidentifikasi komponen elektronika secara presisi tinggi.

## 🎯 2. Sasaran Utama (Objectives)

Penyusunan eksperimen ini diarahkan untuk mencapai beberapa target berikut:

* **Eksplorasi Adaptasi Model:** Memahami mekanisme pemanfaatan ulang bobot model siap pakai (*pretrained weights*) ke ranah spesifik robotika.

* **Otomasi Pengayaan Data:** Mengatasi keterbatasan sampel visual awal melalui teknik *Data Augmentation* secara sistematis.

* **Evaluasi Performa Menyeluruh:** Mengukur tingkat ketepatan prediksi (*accuracy*) serta menganalisis pola kesalahan menggunakan *Confusion Matrix*.

* **Validasi Generalisasi:** Menguji ketahanan model di luar data latih untuk memastikan sistem siap diimplementasikan pada skenario nyata.

## 🧠 3. Paradigma Transfer Learning & Arsitektur

Proyek ini mengandalkan arsitektur **MobileNetV2** sebagai fondasi ekstraksi fitur. Jaringan dasar dibekukan (*frozen layers*) agar pengetahuan visual umumnya tetap terjaga, sementara lapisan atas (*classifier head*) disesuaikan untuk mengenali dua kategori objek output.

```
               [ Pretrained Weights (ImageNet) ]
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    MobileNetV2 Layers   │ (Feature Extractor)
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Custom Dense & Softmax │ (Classification Head)
                 └─────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      [ Kelas MG90S ]                 [ Kelas PCA9685 ]

```

## 📂 4. Tata Letak Direktori Kerja

Struktur direktori dirancang secara modular guna memudahkan pengelolaan dataset mentah, hasil augmentasi, hingga artefak keluaran evaluasi:

```
Transfer Learning/
├── dataset_raw/                  # Kumpulan data hasil augmentasi otomatis
│   ├── servo_driver/
│   └── step_down/
├── foto_asli/                    # Dokumentasi citra referensi utama
│   ├── servo_driver_1.jpg
│   └── step_down_1.jpg
├── results/                      # Artefak evaluasi otomatis
│   ├── accuracy.png              # Grafik histori akurasi
│   ├── loss.png                  # Grafik kurva loss
│   ├── confusion_matrix.png      # Matriks evaluasi kelas
│   └── final_model_comparison.csv# Rekap metrik latihan
├── 1_augmentasi.py               # Skrip generator variasi citra
├── 2_training.py                 # Skrip utama pelatihan & evaluasi model
├── model_modul_elektronik.h5     # Bobot model akhir siap pakai
└── README.md                     # Dokumentasi proyek

```

## ⚡ 5. Panduan Eksekusi Program di VS Code

Untuk mereproduksi eksperimen ini di lingkungan lokal Anda, ikuti tahapan ringkas berikut:

1. **Instalasi Pustaka Pendukung:**
   Buka terminal Visual Studio Code Anda dan jalankan perintah:

   ```
   pip install tensorflow matplotlib pillow scipy scikit-learn seaborn pandas
   
   ```

2. **Eksekusi Tahap Augmentasi Citra:**
   Jalankan skrip pertama untuk memperbanyak variasi data secara sintetis:

   ```
   python 1_augmentasi.py
   
   ```

3. **Inisiasi Pelatihan & Penyimpanan Hasil:**
   Jalankan skrip utama untuk melatih model dan mengekspor grafik performa ke folder `results`:

   ```
   python 2_training.py
   
   ```

## 📈 6. Analisis Hasil & Artefak Evaluasi (`results/`)

Selama proses eksekusi skrip pelatihan, sistem secara otomatis menghasilkan laporan visual dan numerik di dalam folder `results`:

* **`accuracy.png` & `loss.png`**: Menunjukkan tren peningkatan kualitas belajar model seiring bertambahnya iterasi (*epoch*), memastikan tidak terjadi gejala *overfitting* yang ekstrem.

* **`confusion_matrix.png`**: Memberikan gambaran intuitif mengenai sejauh mana model mampu membedakan kedua komponen tanpa adanya kesalahan silang (*misclassification*).

* **`final_model_comparison.csv`**: Menyimpan rekapitulasi data metrik numerik untuk keperluan penyusunan laporan tertulis.

## 🏁 7. Kesimpulan

Penerapan *Transfer Learning* terbukti menjadi solusi paling efektif dan efisien untuk menyelesaikan permasalahan visi komputer pada proyek robotika skala menengah. Dengan dukungan kerangka kerja yang fleksibel serta pemanfaatan model terekstraksi fitur yang matang, sistem mampu mengenali komponen target dengan tingkat keandalan yang tinggi meskipun hanya berbekal sedikit data referensi awal.
