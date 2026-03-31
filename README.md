
# Heart Disease & Stroke Diagnosis with Random Forest

**Course:** Intelligent System (CO5134) — Ho Chi Minh University of Technology (VNU-HCM)  
**Instructor:** Prof. Quan Thanh Tho  
**Students:** Vuong Minh Toan (2491057), Tran Nguyen Huan (2370692)  
**Date:** November 2024  
**Deliverable:** Web app (React + Flask) & ML pipeline for diagnosis support  citeturn2search1
---

## 📌 Mục tiêu
- Xây dựng **mô hình Random Forest** dự đoán **bệnh tim** và **đột quỵ** dựa trên dữ liệu lâm sàng & lối sống.  
- Cung cấp **ứng dụng web** cho cộng đồng: nhập chỉ số sức khỏe → trả về dự đoán + gợi ý cơ bản → tải báo cáo.  citeturn2search1

## 👥 Stakeholders & Lợi ích
- **Người dùng cộng đồng:** tự kiểm tra nguy cơ, nâng cao nhận thức, chi phí thấp.  citeturn2search1
- **Tổ chức cộng đồng/NGO:** dùng cho chiến dịch nâng cao sức khỏe, nhận diện nhóm nguy cơ.  citeturn2search1
- **Chính quyền/Y tế công:** theo dõi xu hướng sức khỏe (ẩn danh), thiết kế can thiệp sớm.  citeturn2search1

## 🧩 Yêu cầu hệ thống
**Chức năng:** nhập dữ liệu cá nhân, hiển thị dự đoán + insight, xuất báo cáo.  citeturn2search1  
**Phi chức năng:** UI trực quan, **thời gian đáp ứng < 5s**, uptime mục tiêu **99.9%**.  citeturn2search1

---

## 🏗️ Kiến trúc & Công nghệ
- **Frontend:** React.js — form nhập liệu, trang kết quả, responsive, tích hợp API qua Axios.  citeturn2search1
- **Backend:** Flask (Python) — nạp mô hình *.pkl*, xử lý dữ liệu, dự đoán, trả kết quả; kiến trúc mô-đun.  citeturn2search1
- **Luồng tổng thể:** người dùng nhập → frontend gọi REST API → backend tiền xử lý & suy luận → trả kết quả → người dùng tải báo cáo.  citeturn2search1
- **Triển khai (gợi ý):** React (Netlify/Vercel/S3), Flask (Gunicorn/uWSGI trên EC2/Heroku/Azure), giám sát Sentry/Prometheus/Grafana, analytics GA.  citeturn2search1

---

## 🗃️ Dữ liệu
- **heart.csv** (≈1025 mẫu) — các đặc trưng: *age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target*.  
  **`target`**: 1 = bệnh tim, 0 = không.  citeturn2search1
- **stroke.csv** (≈5110 mẫu) — các đặc trưng: *gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status, stroke*.  
  **`stroke`**: 1 = có đột quỵ, 0 = không.  citeturn2search1

### Tiền xử lý chính
- **Mã hóa:** Map nhãn nhị phân; **One‑Hot Encoding** cho nhiều lựa chọn (work_type, residence_type, smoking_status…).  citeturn2search1
- **Thiếu dữ liệu:** `bmi` của *stroke.csv* có ~201 giá trị NULL → **drop** vì là biến số và dữ liệu đủ lớn.  citeturn2search1
- **Cân bằng dữ liệu:** *stroke.csv* mất cân bằng mạnh → dùng **SMOTE** để oversample lớp thiểu số.  citeturn2search1
- **Chuẩn hóa & chia tập:** train/test split; chuẩn hóa với `StandardScaler` & `shuffle` khi cần.  citeturn2search1

---

## 🤖 Mô hình & Pipeline
1) **Khảo sát nhanh**: dùng **LazyPredict** để benchmark nhiều mô hình có giám sát → **chọn Random Forest** vì hiệu quả tốt trên cả 2 bài toán.  citeturn2search1  
2) **Huấn luyện:** `RandomForestClassifier(random_state=42)` + tìm tham số tối ưu bằng **GridSearchCV** (`n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features`).  citeturn2search1  
3) **Đánh giá:** Accuracy, Confusion Matrix, Classification Report, **ROC/AUC**.  citeturn2search1

### Kết quả (từ báo cáo)
- **Heart disease:** *Accuracy xấp xỉ 0.99*, **AUC = 1.00** (ROC đạt góc trên‑trái).  citeturn2search1
- **Stroke:** *Accuracy xấp xỉ 0.98*, **AUC = 1.00** (hiệu năng phân biệt rất cao).  citeturn2search1
> Lưu ý: báo cáo có một số số liệu mô tả không hoàn toàn nhất quán ở phần confusion matrix; khuyến nghị chạy lại notebook/skript để in số liệu cuối cùng từ `best_estimator_`.  citeturn2search1

---

## 🔌 API (phác thảo)
- `POST /api/predict/heart` → nhận payload chỉ số sức khỏe → trả dự đoán bệnh tim.  citeturn2search1  
- `POST /api/predict/stroke` → nhận payload chỉ số sức khỏe → trả dự đoán đột quỵ.  citeturn2search1  
- `GET /api/healthz` → kiểm tra tình trạng server.  citeturn2search1  
- `POST /api/feedback` → ghi nhận góp ý người dùng (tùy chọn).  citeturn2search1

> Backend xử lý chuyển đổi kiểu dữ liệu, nạp mô hình *.pkl*, suy luận và trả JSON; frontend hiển thị và cho phép tải báo cáo.  citeturn2search1

---

## 🧪 Cách chạy (gợi ý)
### Backend (Flask)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# đặt model vào ./models/heart_rf.pkl và ./models/stroke_rf.pkl
export FLASK_APP=app.py
flask run --host 0.0.0.0 --port 8000
```

### Frontend (React)
```bash
cd web
npm install
npm run dev  # hoặc npm run build && npm run preview
```

---

## 📂 Gợi ý cấu trúc repo
```bash
intelligent-system-co5134/
├── backend/
│   ├── app.py                # Flask API
│   ├── models/
│   │   ├── heart_rf.pkl
│   │   └── stroke_rf.pkl
│   ├── pipeline/
│   │   ├── preprocess.py     # encode, scale, smote (nếu áp dụng offline)
│   │   └── predict.py
│   ├── requirements.txt
│   └── README_backend.md
├── web/
│   ├── src/                  # React components/forms/pages
│   ├── package.json
│   └── README_frontend.md
├── data/
│   ├── heart.csv
│   └── stroke.csv
├── notebooks/
│   └── training.ipynb        # LazyPredict, GridSearchCV, eval
├── reports/
│   └── Intelligent_System_CO5134.pdf
└── README.md
```

---

## 🔒 Lưu ý đạo đức & pháp lý
- Đây là **công cụ hỗ trợ** (decision support), **không thay thế chẩn đoán y khoa**. Luôn khuyến nghị người dùng **tham vấn bác sĩ**.  citeturn2search1  
- Cần **ẩn danh hóa** và **bảo mật** dữ liệu (tuân thủ quy định địa phương).  citeturn2search1

---

## 🔗 Nguồn mã & Demo
- **Source code:** (điền link Git/Drive khi công bố) — báo cáo hiện ghi **Google Drive**.  citeturn2search1  
- **Demo:** (điền link) — báo cáo hiện ghi **YouTube**.  citeturn2search1

---

## 📣 Gợi ý cải tiến
- Hiệu chỉnh **threshold** theo mục tiêu tăng **recall** (giảm FN) cho ca dương tính.  citeturn2search1  
- Thêm hiệu chỉnh **class weight**, **calibration** (Platt/Isotonic) và **explainability** (SHAP).  
- Log **drift** & giám sát mô hình sau khi triển khai.

---

**© CO5134 — Intelligent System Project**
