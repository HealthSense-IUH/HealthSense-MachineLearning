# Tài liệu tham khảo khoa học - Dự án HealthSense

## 1. Cơ sở lý thuyết HRV (Heart Rate Variability)

| # | Tác giả | Tiêu đề | Tạp chí | Năm | DOI / Link |
|---|---------|---------|---------|-----|------------|
| 1 | Task Force of ESC/NASPE | Heart rate variability: Standards of measurement, physiological interpretation and clinical use | *European Heart Journal*, vol. 17, pp. 354-381 | 1996 | https://doi.org/10.1093/oxfordjournals.eurheartj.a014868 |
| 2 | Shaffer, F. & Ginsberg, J.P. | An Overview of Heart Rate Variability Metrics and Norms | *Frontiers in Public Health*, vol. 5, 258 | 2017 | https://doi.org/10.3389/fpubh.2017.00258 |
| 3 | Malik, M. et al. | Heart rate variability in relation to prognosis after myocardial infarction | *European Heart Journal* | 1989 | https://doi.org/10.1093/oxfordjournals.eurheartj.a059575 |

> **Vai trò:** Định nghĩa 16 đặc trưng HRV (SDNN, RMSSD, pNN50, LF, HF, LF/HF...) và chuẩn đo lường.

---

## 2. Độ tin cậy PPG so với ECG (Chứng minh PPG đủ chính xác)

| # | Tác giả | Tiêu đề | Tạp chí | Năm | DOI / Link | Kết luận chính |
|---|---------|---------|---------|-----|------------|----------------|
| 4 | Schäfer, A. & Vagedes, J. | How accurate is pulse rate variability as an estimate of heart rate variability? A review on studies comparing photoplethysmographic technology with an electrocardiogram | *International Journal of Cardiology*, 166(1), 15-29 | 2013 | https://doi.org/10.1016/j.ijcard.2012.09.119 | PPG và ECG tương quan **r > 0.99** khi nghỉ ngơi |
| 5 | Georgiou, K. et al. | Can Wearable Devices Accurately Measure Heart Rate Variability? A Systematic Review | *Folia Medica*, 60(1), 7-20 | 2018 | https://doi.org/10.2478/folmed-2018-0012 | RMSSD và SDNN sai lệch **< 5%** so với ECG |
| 6 | Bent, B. et al. | Investigating sources of inaccuracy in wearable optical heart rate sensors | *npj Digital Medicine*, 3, 18 | 2020 | https://doi.org/10.1038/s41746-020-0226-6 | Độ chính xác phụ thuộc màu da, vận động, vị trí đeo |
| 7 | Charlot, K. et al. | Interchangeability of HRV measured by PPG and ECG | *Sensors*, 22(13), 4984 | 2022 | https://doi.org/10.3390/s22134984 | ICC > 0.90 khi nghỉ ngơi |

> **Vai trò:** Chứng minh PPG (MAX30102) đủ tin cậy để thay thế ECG trong đo HRV, đặc biệt khi ít vận động.

### Bảng tóm tắt độ tin cậy

| Điều kiện | Tương quan PPG vs ECG | Nguồn |
|-----------|----------------------|-------|
| Ngồi/nằm nghỉ | ICC > 0.90, r > 0.99 | [4][5][7] |
| Đi bộ nhẹ | ICC ~0.85, sai lệch 5-10% | [5][6] |
| Vận động mạnh | ICC giảm, sai lệch 15-20% | [6] |

---

## 3. Phát hiện rung tâm nhĩ (AFib) bằng PPG / Wearable

| # | Tác giả | Tiêu đề | Tạp chí | Năm | DOI / Link | Kết luận chính |
|---|---------|---------|---------|-----|------------|----------------|
| 8 | Perez, M.V. et al. | Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation (Apple Heart Study) | *New England Journal of Medicine*, 381, 1909-1917 | 2019 | https://doi.org/10.1056/NEJMoa1901183 | PPG smartwatch phát hiện AFib với PPV **84%** trên 419,297 người |
| 9 | Bumgarner, J.M. et al. | Smartwatch Algorithm for Automated Detection of Atrial Fibrillation | *Journal of the American College of Cardiology*, 71(21), 2381-2388 | 2018 | https://doi.org/10.1016/j.jacc.2018.03.003 | Sensitivity **93%**, Specificity **84%** |
| 10 | — | Direct-to-Consumer Wearable Smart Devices to Detect Atrial Fibrillation | *JACC: Clinical Electrophysiology* | 2023 | https://doi.org/10.1016/j.jacep.2022.09.011 | So sánh Apple Watch 6, Samsung Galaxy Watch 3 |
| 11 | Düking, P. et al. | Use of a Smart Watch for Early Detection of Paroxysmal Atrial Fibrillation: Validation Study | *JMIR Cardio*, 4(1) | 2020 | https://doi.org/10.2196/14857 | PPG smartwatch phát hiện AFib kịch phát |
| 12 | — | Accuracy of Smartwatches in the Detection of Atrial Fibrillation (Meta-analysis) | *PMC* | 2025 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11737279/ | Sensitivity > 90%, Specificity > 90% |

> **Vai trò:** Chứng minh PPG wearable (giống HealthSense) **đủ khả năng phát hiện AFib** với độ chính xác > 90%.

---

## 4. Dataset công khai cho training

| # | Tên dataset | Nhãn | Nguồn | Link |
|---|-------------|------|-------|------|
| 13 | **MIT-BIH AF Database** | AFib vs Normal (ECG) | PhysioNet | https://physionet.org/content/afdb/1.0.0/ |
| 14 | **PhysioNet CinC 2017** | AFib / Normal / Other / Noisy (ECG) | PhysioNet | https://physionet.org/content/challenge-2017/1.0.0/ |
| 15 | **PPG-DaLiA** | 8 hoạt động (PPG) | UCI ML Repository | https://archive.ics.uci.edu/dataset/495/ppg+dalia |

> **Vai trò:** MIT-BIH AF hoặc CinC 2017 dùng để train model. ECG → RR intervals → HRV features → cùng feature space với PPG.

---

## 5. Xử lý tín hiệu PPG

| # | Tác giả | Tiêu đề | Tạp chí | Năm | DOI / Link |
|---|---------|---------|---------|-----|------------|
| 16 | Reiss, A. et al. | Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks | *Sensors*, 19(14), 3079 | 2019 | https://doi.org/10.3390/s19143079 |
| 17 | Biswas, D. et al. | CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment | *IEEE Transactions on Instrumentation and Measurement* | 2019 | https://doi.org/10.1109/TIM.2018.2882523 |

> **Vai trò:** Phương pháp bandpass filter, peak detection, motion artifact compensation cho PPG.

---

## 6. Chuỗi logic chứng minh cho đồ án

```
Bước 1: HRV là chỉ số y khoa đáng tin cậy
         → [1] Task Force 1996, [2] Shaffer 2017

Bước 2: PPG đo HRV chính xác gần bằng ECG (r > 0.99 khi nghỉ)
         → [4] Schäfer 2013, [5] Georgiou 2018, [7] Charlot 2022

Bước 3: PPG wearable phát hiện AFib với Sensitivity > 90%
         → [8] Apple Heart Study 2019, [9] Bumgarner 2018, [12] Meta-analysis 2025

Bước 4: Train model từ ECG dataset (MIT-BIH) áp dụng cho PPG (MAX30102)
         Vì ECG và PPG → cùng RR intervals → cùng HRV features
         → [4] Schäfer 2013, [16] Reiss 2019

Bước 5: Normalization (Z-score) xử lý domain shift giữa thiết bị
         → [16] Reiss 2019, [17] Biswas 2019
```
