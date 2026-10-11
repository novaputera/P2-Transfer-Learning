# 🤖 Proyek Klasifikasi Komponen Elektronika Berbasis Transfer Learning (RET503)

Repositori ini mendokumentasikan implementasi *Deep Learning* dan *Transfer Learning* untuk pengenalan citra objek modular robotik, khususnya dalam membedakan antara modul **Step-Down (Buck Converter)** dan **Servo Driver (PCA9685)**. Proyek ini dikembangkan untuk memenuhi tugas mata kuliah **RET503 Computer Vision and Deep Learning** di Program Studi Teknologi Rekayasa Robotika, Politeknik Negeri Batam.

---

## 🚀 Gambaran Umum Proyek

Sistem visi komputer pada robot modern menuntut tingkat ketepatan dan efisiensi yang tinggi. Alih-alih merancang dan melatih model jaringan saraf dari awal yang memerlukan waktu komputasi besar serta dataset berjumlah ribuan, pendekatan *Transfer Learning* memanfaatkan model pra-latih (*pretrained backbone*) yang telah teruji untuk mengenali fitur visual mendasar dengan sangat baik.

---

## 📂 Struktur Direktori Proyek

```text
Transfer Learning/
├── dataset_raw/                  # Kumpulan citra hasil augmentasi otomatis per kelas
│   ├── servo_driver/
│   └── step_down/
├── foto_asli/                    # Citra referensi utama sumber augmentasi
│   ├── servo_driver_1.jpg
│   └── step_down_1.jpg
├── results/                      # Artefak hasil pelatihan & evaluasi otomatis
│   ├── accuracy.png              # Grafik kurva tingkat akurasi
│   ├── loss.png                  # Grafik kurva fungsi kerugian (loss)
│   ├── confusion_matrix.png      # Matriks kebingungan model (evaluasi kelas)
│   └── final_model_comparison.csv# Rekapitulasi metrik numerik pelatihan
├── 1_augmentasi.py               # Skrip generator variasi data citra sintetis
├── 2_training.py                 # Skrip utama pelatihan model & evaluasi kinerja
├── model_modul_elektronik.h5     # Berkas bobot arsitektur model siap pakai
└── README.md                     # Berkas dokumentasi utama repositori
```

---

## ⚙️ Panduan Instalasi & Eksekusi

Ikuti langkah-langkah berikut untuk menjalankan eksperimen ini di komputer Anda:

1. **Clone atau Buka Proyek di VS Code:**
   Pastikan Anda membuka direktori utama proyek (`C:\Transfer Learning`) melalui Visual Studio Code.

2. **Instalasi Pustaka Pendukung:**
   Buka terminal di VS Code dan jalankan perintah berikut untuk menginstal seluruh dependensi yang diperlukan:
   ```bash
   pip install tensorflow matplotlib pillow scipy scikit-learn seaborn pandas
   ```

3. **Jalankan Proses Augmentasi Citra:**
   Eksekusi skrip pertama untuk memperbanyak sampel data secara otomatis dari foto referensi:
   ```bash
   python 1_augmentasi.py
   ```

4. **Laksanakan Proses Pelatihan Model:**
   Jalankan skrip kedua untuk mulai melatih model, mengevaluasi data, serta mengekspor grafik performa ke dalam folder `results`:
   ```bash
   python 2_training.py
   ```

---

## 📊 Artefak Evaluasi & Hasil (`results/`)

Setiap kali proses pelatihan selesai, sistem secara otomatis merangkum hasil evaluasi ke dalam direktori `results`:
* **Grafik Akurasi & Loss (`accuracy.png`, `loss.png`):** Membantu memantau kestabilan proses belajar model pada setiap iterasi (*epoch*).
* **Confusion Matrix (`confusion_matrix.png`):** Memvisualisasikan tingkat keberhasilan model dalam memisahkan kedua kategori modul tanpa ada kesalahan silang.
* **Laporan Metrik (`final_model_comparison.csv`):** Rekap data numerik untuk keperluan analisis lanjutan pada laporan praktikum.