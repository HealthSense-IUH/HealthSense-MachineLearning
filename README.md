# HealthSense ML

## Tiếng Việt

**HealthSense ML** là kho chứa mã nguồn dành riêng cho việc phân tích dữ liệu và huấn luyện mô hình Machine Learning cho dự án HealthSense.

### Chức năng chính
- Thu thập và lưu trữ dữ liệu PPG thô từ cảm biến MAX30102 qua ESP32.
- Tiền xử lý tín hiệu: loại bỏ nhiễu chuyển động (Motion Artifact) và Baseline Wander.
- Trích xuất 16 đặc trưng HRV (Heart Rate Variability) theo chuẩn Task Force 1996.
- Huấn luyện mô hình phân loại trạng thái sức khỏe bằng Random Forest.

### Công nghệ
- **Ngôn ngữ:** Python 3.12+
- **Xử lý tín hiệu:** SciPy (Butterworth Filter, Welch Periodogram)
- **Phân tích dữ liệu:** Pandas, NumPy, Matplotlib
- **Machine Learning:** Scikit-learn
- **Môi trường thí nghiệm:** Jupyter Notebook

### Cấu trúc dự án
- `data/raw/`: Chứa các file CSV gốc thu thập từ ESP32.
- `data/processed/`: Chứa các file CSV sau khi đã làm sạch và lọc nhiễu.
- `data/features/`: Chứa bảng đặc trưng HRV đã trích xuất (16 features).
- `notebooks/01_preprocessing.ipynb`: Tiền xử lý tín hiệu PPG (Bandpass Filter, Peak Detection).
- `notebooks/02_feature_engineering.ipynb`: Trích xuất 16 đặc trưng HRV (SDNN, RMSSD, LF/HF, ...).
- `notebooks/03_model_training.ipynb`: Huấn luyện và đánh giá mô hình Random Forest.
- `models/`: Chứa file mô hình đã huấn luyện (model.pkl).

### Cài đặt và Sử dụng
1. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Mở Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Chạy lần lượt các notebook từ `01_preprocessing` đến `03_model_training`.

### Tài liệu tham khảo
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Malik et al., "Heart rate variability in relation to prognosis after myocardial infarction," *European Heart Journal*, 1989.
- [3] Shaffer, F. & Ginsberg, J.P., "An Overview of Heart Rate Variability Metrics and Norms," *Frontiers in Public Health*, vol. 5, 2017.

---

## English

**HealthSense ML** is the repository dedicated to data analysis and Machine Learning model training for the HealthSense project.

### Key Features
- Collects and stores raw PPG data from the MAX30102 sensor via ESP32.
- Signal preprocessing: removes Motion Artifacts and Baseline Wander.
- Extracts 16 HRV (Heart Rate Variability) features following the Task Force 1996 standard.
- Trains a health status classification model using Random Forest.

### Tech Stack
- **Language:** Python 3.12+
- **Signal Processing:** SciPy (Butterworth Filter, Welch Periodogram)
- **Data Analysis:** Pandas, NumPy, Matplotlib
- **Machine Learning:** Scikit-learn
- **Experimentation:** Jupyter Notebook

### Project Structure
- `data/raw/`: Raw CSV files collected from ESP32.
- `data/processed/`: Cleaned and filtered CSV files.
- `data/features/`: Extracted HRV feature tables (16 features).
- `notebooks/01_preprocessing.ipynb`: PPG signal preprocessing (Bandpass Filter, Peak Detection).
- `notebooks/02_feature_engineering.ipynb`: 16 HRV feature extraction (SDNN, RMSSD, LF/HF, ...).
- `notebooks/03_model_training.ipynb`: Random Forest model training and evaluation.
- `models/`: Trained model files (model.pkl).

### Installation and Usage
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run the notebooks sequentially from `01_preprocessing` to `03_model_training`.

### References
- [1] Task Force of ESC/NASPE, "Heart rate variability: Standards of measurement, physiological interpretation and clinical use," *European Heart Journal*, vol. 17, pp. 354-381, 1996.
- [2] Malik et al., "Heart rate variability in relation to prognosis after myocardial infarction," *European Heart Journal*, 1989.
- [3] Shaffer, F. & Ginsberg, J.P., "An Overview of Heart Rate Variability Metrics and Norms," *Frontiers in Public Health*, vol. 5, 2017.
