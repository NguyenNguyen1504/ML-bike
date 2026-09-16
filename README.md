
# ML-bike

Dự án phân tích và huấn luyện mô hình dự đoán xe đạp (Machine Learning).

## Dữ liệu

- `data/YYYY-MM.csv`: dữ liệu chuyến đi xe đạp công cộng Helsinki & Espoo (mùa tháng 4–tháng 10, 2022–2025).
- `data/weather_daily.csv`: thời tiết theo ngày lấy từ FMI open data (trạm Helsinki Kaisaniemi, `fmisid=100971`),
  gồm nhiệt độ trung bình ngày (`temp_c`) và lượng mưa ngày (`precip_mm`).
  FMI dùng giá trị `-1` cho ngày không đo được lượng mưa; notebook quy các giá trị này về 0.
  Nếu xoá file này, notebook sẽ tự tải lại từ FMI (cần internet và gói `certifi`).

## Hướng dẫn chạy nhanh

### 1. Cài đặt môi trường ảo

Mở terminal tại thư mục dự án và chạy:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường (macOS / Linux)
source .venv/bin/activate

# Kích hoạt trên Windows (nếu dùng PowerShell / CMD)
.venv\Scripts\activate

```

### 2. Cài đặt thư viện

```bash
# Nâng cấp pip
pip install --upgrade pip

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt

```

### 3. Chạy Notebook

1. Mở file `ML_bike.ipynb` trong VS Code.
2. Nhấn vào mục **Select Kernel** ở góc trên cùng bên phải.
3. Chọn Python Environments rồi trỏ tới kernel `.venv` vừa tạo để chạy các cell code.

