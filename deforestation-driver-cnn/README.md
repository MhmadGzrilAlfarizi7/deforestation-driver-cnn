# Klasifikasi Driver Deforestasi dengan CNN Multi-Sensor (Sentinel-1 + Sentinel-2)

Studi kasus dari kuliah tamu **Dr. Bart Slachter (Wageningen University)** tentang *deforestation alerts* dan
atribusi *direct driver* menggunakan citra satelit dan deep learning.

## Latar belakang
Sistem alert deforestasi hanya memberi tahu *di mana* hutan berubah, bukan *penyebabnya*. Pemateri memakai CNN
yang menerima tumpukan citra radar (Sentinel-1) dan optik (Sentinel-2) untuk mengklasifikasikan 7 driver:
smallholder agriculture, large-scale agriculture, road, selective logging, mining, wildfire, natural disturbance.

## Pertanyaan studi kasus
1. Apakah CNN lebih baik daripada Random Forest (baseline konvensional)? *(rekomendasi pemateri: selalu bandingkan dengan metode konvensional)*
2. Apakah fusi S1+S2 membuat model tahan terhadap tutupan awan?
3. Seberapa besar selisih **accuracy** vs **macro-F1** pada data tak seimbang? *(accuracy didominasi kelas agriculture)*

## Data
Data **sintetis** (`src/data.py`): patch 16x16 dengan 6 channel (S2: B2,B3,B4,B8; S1: VV,VH), kelas tak seimbang
(agriculture dominan, mining ~3%), noise, dan simulasi awan yang merusak channel optik. Ini simulasi untuk
latihan, bukan data lapangan, jadi angka hasilnya tidak bisa diklaim sebagai performa di dunia nyata.

## Cara menjalankan
```bash
pip install -r requirements.txt
cd src
python train.py --n 6000 --epochs 15
```
Output: tabel accuracy dan macro-F1 untuk RF, CNN S1+S2, CNN S2 saja, CNN S1 saja pada kondisi cerah vs berawan,
disimpan di `results/metrics.json`.

## Menggunakan data nyata
Ganti `make_dataset` dengan loader patch Sentinel yang sudah berlabel (array `(N, 6, H, W)` + label 0-6).
Sumber alert: Global Nature Watch (RADD / GLAD); citra: Copernicus Data Space atau Google Earth Engine.
Label ground truth perlu dibuat/diverifikasi manual (poin penting dari sesi Q&A).

## Keterbatasan (dari materi)
- Hanya *direct driver*, penyebab sebenarnya bisa baru terlihat kemudian.
- Wildfire vs flooding sering tertukar (sering terjadi di dataran banjir sungai).
- Satelit saja tidak cukup untuk nuansa seperti penebangan artisanal atau arang; perlu data lapangan.
