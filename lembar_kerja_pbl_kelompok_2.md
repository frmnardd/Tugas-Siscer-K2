# Lembar Kerja PBL Pertemuan 10: Data Preprocessing & Feature Engineering
**Kelompok:** Kelompok 2  
**Studi Kasus:** Memprediksi Mahasiswa yang Lulus Tepat Waktu (`data_akademik_kelulusan.csv`)

## 1. Model Pertama (Sebelum Preprocessing)

### Hasil Eksekusi Sel 1–2
- **Ukuran data awal (`df.shape`):** `(312, 14)`
- **Akurasi model pertama (`model0`):** `0.619` (61,9%)
- **Akurasi jika selalu menebak `"Ya"` (`y_te.mean()`):** `0.587` (58,7%)

### Jawaban Pertanyaan 1
**Pertanyaan:** Bandingkan akurasi model pertama dengan akurasi kalau kita selalu menebak `"Ya"`. Apa artinya?

**Jawaban:**
- Akurasi model pertama (**0.619** atau **61,9%**) hanya lebih tinggi **0.032** (**3,2 poin persentase**) dibandingkan tebakan naif yang selalu menebak kelas mayoritas `"Ya"` (**0.587** atau **58,7%**).
- Artinya, **model pertama belum berhasil mempelajari pola akademik yang bermakna** dan kinerjanya hampir setara dengan menebak secara buta. Penyebab utamanya adalah:
  1. Kolom `nim` (angka identitas berukuran $\sim 2,4 \times 10^{13}$) dan `penghasilan_ortu` (jutaan rupiah) langsung dimasukkan tanpa *feature selection* dan tanpa *scaling*, sehingga mendominasi perhitungan jarak Euclidean pada algoritma KNN.
  2. Nilai kosong (`NaN`) diisi langsung dengan `0` (`fillna(0)`), yang mendistorsi distribusi fitur (misalnya IPK kosong dianggap IPK `0.0`).
  3. Baris duplikat dan nilai salah ketik (*outlier* tidak masuk akal) belum dibersihkan, serta kolom kategorikal (`prodi`, `jalur_masuk`, `ikut_organisasi`, `kemampuan_bahasa`) diabaikan.

## 2. Eksplorasi dan Temuan Masalah di Data

### Kode Lengkap Sel Eksplorasi (`TODO` Bagian 2)
```python
print("Jumlah nilai kosong per kolom:")
print(df.isna().sum())
print("\nJumlah baris ganda:", df.duplicated().sum())  # TODO dilengkapi
```

### Hasil Eksekusi Eksplorasi Data
- **Jumlah baris ganda (`df.duplicated().sum()`):** `12` baris duplikat identik (dari 312 baris, terdapat 300 mahasiswa unik).
- **Jumlah nilai kosong (`NaN`) per kolom:**
  - `kehadiran`: **22** nilai kosong
  - `jam_belajar`: **19** nilai kosong
  - `penghasilan_ortu`: **16** nilai kosong
  - `ipk_sem4`: **9** nilai kosong
  - `kemampuan_bahasa`: **6** nilai kosong
  - Kolom lainnya (`nim`, `jenis_kelamin`, `prodi`, `jalur_masuk`, `kampus`, `sks_lulus`, `nilai_ta`, `lulus_tepat_waktu`): **0** nilai kosong.
- **Distribusi `jenis_kelamin` sebelum pembersihan:** `L` (150), `P` (119), `perempuan ` (10), `Perempuan` (8), `laki` (8), `l` (8), `Laki-laki` (6), `p` (3).
- **Distribusi `prodi` sebelum pembersihan:** `Informatika` (97), `Sistem Informasi` (81), `Statistika` (67), `Matematika` (52), `matematika` (6), `sistem informasi` (5), `statistika` (2), `informatika` (2).
- **Distribusi `kampus`:** `Tembalang` (312 baris — seluruh data bernilai tunggal).

### Jawaban Pertanyaan 2
**Pertanyaan:** Tuliskan minimal lima masalah yang Anda temukan, beserta kolomnya.

**Jawaban:**
1. **Nilai kosong (*Missing Values*)** pada 5 kolom: `kehadiran` (22 kosong), `jam_belajar` (19 kosong), `penghasilan_ortu` (16 kosong), `ipk_sem4` (9 kosong), dan `kemampuan_bahasa` (6 kosong).
2. **Data ganda (*Duplicate Rows*)** pada tabel: terdapat **12 baris duplikat** identik (312 baris untuk 300 NIM unik).
3. **Penulisan kategori tidak konsisten pada kolom `jenis_kelamin`**: terdapat 8 variasi penulisan untuk 2 nilai yang sama (`"L"`, `"P"`, `"perempuan "` dengan spasi di akhir, `"Perempuan"`, `"laki"`, `"l"`, `"Laki-laki"`, dan `"p"`).
4. **Kapitalisasi tidak konsisten pada kolom `prodi`**: nama program studi terpecah antara huruf besar dan huruf kecil (`"Informatika"` vs `"informatika"`, `"Sistem Informasi"` vs `"sistem informasi"`, `"Statistika"` vs `"statistika"`, `"Matematika"` vs `"matematika"`).
5. **Salah ketik desimal (*outlier* tidak valid) pada kolom `ipk_sem4`**: terdapat nilai IPK di atas skala maksimum 4,00 yaitu **`28.3`, `32.8`, dan `37.3`** (seharusnya `2.83`, `3.28`, dan `3.73`).
6. **Nilai tidak mungkin (*impossible values*) pada kolom `kehadiran`**: persentase kehadiran mencapai **`850.0` dan `920.0`**, melampaui batas maksimum `100%`.
7. **Nilai tidak mungkin (*impossible values*) pada kolom `jam_belajar`**: terdapat angka **`120.0` dan `168.0` jam per minggu**, padahal total jam dalam 1 minggu penuh hanya 168 jam.
8. **Kolom tidak relevan, konstan, dan bocor (*data leakage*)**: `nim` hanya berupa nomor identitas mahasiswa, `kampus` bernilai tunggal (`"Tembalang"` di semua baris / varians nol), dan `nilai_ta` baru diketahui saat mahasiswa menyelesaikan Tugas Akhir menjelang lulus (*data leakage*).

## 3. Data Cleaning & Penanganan Outlier

### Kode Lengkap Sel 3a & 3b (`TODO` Bagian 3)
```python
data = df.copy()

# 3a. Hapus baris ganda
data = data.drop_duplicates()  # TODO dilengkapi

# 3b. Seragamkan penulisan jenis kelamin dan prodi
peta_jk = {"l": "L", "laki-laki": "L", "laki": "L", "p": "P", "perempuan": "P"}
data["jenis_kelamin"] = data["jenis_kelamin"].str.strip().str.lower().map(peta_jk)
data["prodi"] = data["prodi"].str.strip().str.title()

print(len(data), "baris")
print(data["jenis_kelamin"].value_counts())
```
- **Output:** `300 baris` dengan distribusi `jenis_kelamin`: `L = 165`, `P = 135`.

### Hasil Pemeriksaan IQR (`cek_iqr`)
- `ipk_sem4`: batas `2.03` sampai `4.24`, ditandai **5 nilai** $\rightarrow$ `[2.01, 28.3, 32.8, 37.3, 37.3]`
- `kehadiran`: batas `57.50` sampai `109.50`, ditandai **2 nilai** $\rightarrow$ `[850.0, 920.0]`
- `jam_belajar`: batas `-4.50` sampai `23.50`, ditandai **7 nilai** $\rightarrow$ `[24.0, 24.0, 25.0, 32.0, 120.0, 120.0, 168.0]`

### Jawaban Pertanyaan 3
**Pertanyaan:** Dari nilai yang ditandai, mana yang jelas salah input dan mana yang masih masuk akal?

**Jawaban:**
- **Nilai yang jelas salah input (*Data Error / Invalid*):**
  - Pada `ipk_sem4`: nilai **`28.3`, `32.8`, `37.3`, dan `37.3`** jelas salah ketik titik desimal karena skala IPK maksimum adalah `4.00` (seharusnya `2.83`, `3.28`, dan `3.73`).
  - Pada `kehadiran`: nilai **`850.0` dan `920.0`** jelas salah input karena persentase kehadiran kuliah tidak mungkin lebih dari `100%`.
  - Pada `jam_belajar`: nilai **`120.0`, `120.0`, dan `168.0`** jam per minggu jelas tidak mungkin karena 1 minggu hanya memiliki total 168 jam (batas wajar maksimal adalah 84 jam/minggu atau 12 jam/hari).
- **Nilai yang masih masuk akal (*Nilai Ekstrem Asli / Valid*):**
  - Pada `ipk_sem4`: nilai **`2.01`** masih berada dalam rentang sah IPK (`0.00–4.00`), sehingga merupakan data mahasiswa nyata dengan IPK rendah dan tidak boleh dibuang.
  - Pada `jam_belajar`: nilai **`24.0`, `24.0`, `25.0`, dan `32.0`** jam per minggu (sekitar 3,4 hingga 4,5 jam belajar mandiri per hari) sangat masuk akal bagi mahasiswa yang rajin, sehingga harus dipertahankan.

### Kode Lengkap Perbaikan Outlier (`TODO` Bagian 3)
```python
# IPK di atas 4 adalah salah ketik (3.73 tertulis 37.3): bagi 10
salah_ipk = data["ipk_sem4"] > 4
data.loc[salah_ipk, "ipk_sem4"] = data.loc[salah_ipk, "ipk_sem4"] / 10

# Kehadiran di atas 100% dan jam belajar di atas 84 jam per minggu tidak mungkin: jadikan kosong
# TODO dilengkapi:
data.loc[data["kehadiran"] > 100, "kehadiran"] = np.nan
data.loc[data["jam_belajar"] > 84, "jam_belajar"] = np.nan

data[["ipk_sem4", "kehadiran", "jam_belajar"]].describe().round(2)
```

### Statistik Deskriptif Setelah Perbaikan Outlier
| Statistik | `ipk_sem4` | `kehadiran` | `jam_belajar` |
|---|---:|---:|---:|
| **count** | 292.00 | 276.00 | 279.00 |
| **mean** | 3.13 | 82.92 | 10.03 |
| **std** | 0.38 | 8.75 | 5.32 |
| **min** | 2.01 | 63.00 | 0.00 |
| **25%** | 2.86 | 77.00 | 6.00 |
| **50% (median)** | 3.13 | 83.00 | 9.00 |
| **75%** | 3.38 | 90.00 | 13.00 |
| **max** | 4.00 | 100.00 | 32.00 |

## 4. Feature Selection

### Kode Lengkap (`TODO` Bagian 4)
```python
kolom_dibuang = ["nim", "kampus", "nilai_ta"]  # TODO dilengkapi
kolom_numerik = ["ipk_sem4", "sks_lulus", "kehadiran", "jam_belajar", "penghasilan_ortu"]
kolom_nominal = ["jenis_kelamin", "prodi", "jalur_masuk", "ikut_organisasi"]
kolom_ordinal = ["kemampuan_bahasa"]

print("Kolom yang dibuang:", kolom_dibuang)
X = data[kolom_numerik + kolom_nominal + kolom_ordinal]
y = (data["lulus_tepat_waktu"] == "Ya").astype(int)
print(X.shape)
```
- **Output:** `Kolom yang dibuang: ['nim', 'kampus', 'nilai_ta']`, ukuran `X.shape` adalah `(300, 10)`.

### Jawaban Pertanyaan 4
**Pertanyaan:** Jelaskan alasan setiap kolom yang dibuang. Mengapa `nilai_ta` berbahaya walaupun sangat berkaitan dengan label?

**Jawaban:**
1. **`nim`**: Merupakan nomor identitas unik mahasiswa (*identifier* / ID) yang tidak memiliki hubungan kausal maupun pola prediktif terhadap kelulusan; jika dimasukkan sebagai angka, nilainya yang mencapai puluhan triliun ($\sim 2,4 \times 10^{13}$) akan mengacaukan perhitungan jarak dan bobot fitur.
2. **`kampus`**: Memiliki nilai yang sama persis di seluruh baris (`"Tembalang"`, varians = 0 / *constant feature*), sehingga tidak memberikan informasi pembeda apa pun bagi model.
3. **`nilai_ta` (Mengapa berbahaya?)**: Menimbulkan **kebocoran data (*data leakage* / *target leakage*)**. Tujuan pembuatan model oleh bagian akademik adalah memprediksi risiko kelulusan sejak dini (misalnya saat mahasiswa berada di **semester 4**), sedangkan `nilai_ta` (nilai Tugas Akhir/Skripsi) baru diketahui di semester akhir saat mahasiswa akan lulus. Meskipun sangat berkaitan dengan label `lulus_tepat_waktu`, fitur ini **belum tersedia di dunia nyata pada saat prediksi dilakukan di semester 4**.

## 5. Train-Test Split

### Kode Lengkap (`TODO` Bagian 5)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # TODO dilengkapi: stratify=y
print(X_train.shape, X_test.shape)
```
- **Output:** `(240, 10) (60, 10)` (240 baris data *training* dan 60 baris data *testing* dengan proporsi label `y` terjaga seimbang).

## 6. Pipeline Preprocessing

### Kode Lengkap (`TODO` Bagian 6)
```python
pra_numerik = Pipeline([
    ("imputasi", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler()),  # TODO dilengkapi
])
pra_nominal = Pipeline([
    ("imputasi", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(handle_unknown="ignore")),  # TODO dilengkapi
])
pra_ordinal = Pipeline([
    ("imputasi", SimpleImputer(strategy="most_frequent")),
    ("encoding", OrdinalEncoder(categories=[["Dasar", "Menengah", "Mahir"]])),
])
preprocessing = ColumnTransformer([
    ("num", pra_numerik, kolom_numerik),
    ("nom", pra_nominal, kolom_nominal),
    ("ord", pra_ordinal, kolom_ordinal),
])
```

## 7. Evaluasi Model Supervised Sederhana

### Hasil Akurasi Model (Bagian 7)
| Model | Akurasi Test | Selisih terhadap Model Pertama |
|---|---:|---:|
| **KNN (dengan Pipeline Preprocessing)** | **0.833** (83,3%) | **+0.214** (+21,4 poin persentase) |
| **Regresi Logistik (dengan Pipeline Preprocessing)** | **0.800** (80,0%) | **+0.181** (+18,1 poin persentase) |
| **Model Pertama (Sebelum Preprocessing)** | **0.619** (61,9%) | Baseline awal |

### Jawaban Pertanyaan 5
**Pertanyaan:** Berapa kenaikan akurasi dibanding model pertama?

**Jawaban:**
- Pada model **KNN**, akurasi meningkat dari **0.619 (61,9%)** menjadi **0.833 (83,3%)**, yaitu mengalami kenaikan sebesar **+0.214 (+21,4 poin persentase)**.
- Pada model **Regresi Logistik**, akurasi meningkat dari **0.619 (61,9%)** menjadi **0.800 (80,0%)**, yaitu mengalami kenaikan sebesar **+0.181 (+18,1 poin persentase)**.

## 8. Eksperimen Tambahan

### 8a. KNN Tanpa Scaling
- **Output eksekusi:** `KNN tanpa scaling: 0.617` (61,7%)

#### Jawaban Pertanyaan 6
**Pertanyaan:** Mengapa KNN sangat terpengaruh oleh scaling? Kolom mana yang mendominasi perhitungan jarak?

**Jawaban:**
- **Mengapa KNN sangat terpengaruh oleh *scaling*:** Algoritma KNN menentukan ketetanggaan antar mahasiswa menggunakan jarak geometris (seperti jarak Euclidean $\sqrt{\sum (x_i - y_i)^2}$) di seluruh fitur. Apabila skala antar fitur tidak disetarakan, fitur yang memiliki satuan angka sangat besar akan menghasilkan selisih kuadrat yang raksasa dan menenggelamkan kontribusi fitur lain yang berskala kecil (seperti `ipk_sem4` yang hanya berkisar `2.01–4.00` atau fitur hasil *encoding* `0–2`).
- **Kolom yang mendominasi jarak:** Kolom **`penghasilan_ortu`**, karena nilainya berada dalam jutaan hingga puluhan juta rupiah (`Rp1.200.000` – `Rp22.400.000`), sedangkan fitur akademik penting lainnya hanya bernilai puluhan (`sks_lulus`, `kehadiran`, `jam_belajar`) atau satuan (`ipk_sem4`). Tanpa *scaling*, KNN praktis hanya mencari tetangga terdekat berdasarkan kemiripan penghasilan orang tua, sehingga akurasinya anjlok dari **0.833** menjadi **0.617**.

### 8b. Bagaimana Jika `nilai_ta` Ikut Dipakai?
- **Output eksekusi:** `Akurasi dengan nilai_ta: 1.0` (100%)

#### Jawaban Pertanyaan 7
**Pertanyaan:** Akurasinya sangat tinggi. Mengapa hasil ini tidak bisa dipakai untuk memprediksi mahasiswa semester 4?

**Jawaban:**
- Meskipun menghasilkan akurasi **1.0 (100%)** pada data historis, model ini mengalami **kebocoran data (*data leakage*)** dan tidak dapat digunakan untuk memprediksi mahasiswa semester 4 karena **mahasiswa semester 4 belum mengambil atau menyelesaikan Tugas Akhir/Skripsi, sehingga `nilai_ta` belum ada pada saat prediksi dilakukan**. Model yang bergantung pada informasi dari masa depan tidak dapat dijalankan pada mahasiswa aktif semester 4 untuk keperluan intervensi akademik dini.

## 9. Ringkasan untuk Lembar Kerja (Kelompok 2)

| Langkah | Tindakan Kelompok 2 |
|---|---|
| **Cleaning** | Menghapus 12 baris ganda (`data.drop_duplicates()` $\rightarrow$ 300 baris unik); menyeragamkan penulisan `jenis_kelamin` menjadi `L`/`P` dan kapitalisasi `prodi` (`str.title()`); memperbaiki salah ketik `ipk_sem4 > 4` dengan dibagi 10; mengubah nilai tidak mungkin (`kehadiran > 100` dan `jam_belajar > 84`) menjadi `NaN`; serta mengimputasi nilai kosong dengan `median` (fitur numerik) dan `most_frequent`/modus (fitur kategorikal). |
| **Scaling** | Menerapkan `StandardScaler()` pada 5 kolom numerik (`ipk_sem4`, `sks_lulus`, `kehadiran`, `jam_belajar`, `penghasilan_ortu`) di dalam `Pipeline` agar setiap fitur numerik memiliki rata-rata 0 dan simpangan baku 1 sehingga `penghasilan_ortu` tidak mendominasi perhitungan jarak. |
| **Encoding** | Menerapkan `OneHotEncoder(handle_unknown="ignore")` pada 4 kolom kategorikal nominal (`jenis_kelamin`, `prodi`, `jalur_masuk`, `ikut_organisasi`) dan `OrdinalEncoder(categories=[["Dasar", "Menengah", "Mahir"]])` pada kolom kategorikal bertingkat/ordinal (`kemampuan_bahasa`). |
| **Feature yang dibuang** | Membuang 3 kolom: `nim` (nomor identitas unik tanpa makna prediktif), `kampus` (bernilai konstan `"Tembalang"` pada seluruh baris / varians nol), dan `nilai_ta` (menyebabkan *data leakage* karena baru diketahui di akhir studi, bukan di semester 4). |
| **Pembagian data** | Membagi data menggunakan `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` menghasilkan 240 data *training* dan 60 data *testing* **sebelum** imputasi, *scaling*, dan *encoding* agar distribusi label seimbang dan tidak terjadi kebocoran informasi (*data leakage*) dari data *test* ke *training*. |
| **Akurasi sebelum → sesudah** | **0.619 (61,9%)** pada model pertama $\rightarrow$ meningkat menjadi **0.833 (83,3%)** pada KNN (**+21,4 poin persentase**) dan **0.800 (80,0%)** pada Regresi Logistik (**+18,1 poin persentase**). *(Eksperimen tambahan: KNN tanpa scaling turun ke **0.617**, sedangkan model bocor dengan `nilai_ta` mencapai **1.000**)*. |
