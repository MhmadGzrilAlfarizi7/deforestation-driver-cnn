# Ringkasan Sesi: Deforestation Alerts & AI Driver Attribution (Dr. Bart Slachter)

- Hutan tropis penting untuk karbon, siklus air, biodiversitas, dan mata pencaharian; hilang karena alih guna lahan, logging, tambang.
- Sentinel-2 (optik): informasi vegetasi, tapi terhalang awan. Sentinel-1 (radar): tembus awan, lebih sulit diinterpretasi.
- Alert deforestasi: mingguan, resolusi 10-30 m (contoh RADD berbasis Sentinel-1). Kelemahan: tidak menjelaskan penyebab.
- CNN dengan input stack S1 (2 band) + S2 (4 band) mengklasifikasikan 7 driver; model belajar sendiri mengandalkan radar saat optik berawan.
- Pertanian dipecah menurut ukuran (smallholder vs large scale) dan keberadaan api.
- Temuan: Amerika Selatan didominasi large-scale agri + wildfire (kekeringan 2024); Afrika Tengah smallholder + logging; Asia Tenggara campuran agri, logging, tambang.
- Akurasi: overall tinggi (didominasi agriculture), macro-F1 lebih rendah; wildfire dan natural disturbance paling sulit.
- Keterbatasan: konsistensi satelit, hanya direct driver, latensi bulanan, perlu data lapangan.
- Q&A: mulai dari ML konvensional sebagai benchmark; S1 lebih baik untuk banjir; untuk api realtime pakai sensor termal; kolaborasi butuh data ground truth.
