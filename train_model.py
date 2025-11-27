#!/usr/bin/env python
"""
Huấn luyện mô hình Isolation Forest trên dữ liệu log.

Script này đọc một file CSV chứa log web với các cột:
  - timestamp: thời gian xảy ra request ở định dạng yyyy-MM-dd HH:mm:ss
  - src_ip: địa chỉ IP nguồn
  - dst_ip: địa chỉ IP đích
  - method: phương thức HTTP (GET, POST,...)
  - url: đường dẫn URL
  - status: mã trạng thái HTTP (200, 404, 500,...)
  - response_time: thời gian phản hồi (giây)
  - bytes: số byte trả về

Chúng ta chọn ba trường số (status, response_time, bytes) làm đặc trưng,
chuẩn hóa bằng StandardScaler và huấn luyện Isolation Forest.  Kết quả gồm
model và scaler được lưu vào file bằng joblib.

Tham số "contamination" xác định tỷ lệ phần trăm mẫu được xem là bất thường.

Ví dụ sử dụng:
    python train_model.py --file data/sample_logs.csv --model-file model.pkl --contamination 0.01
"""

import argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


def train_model(file_path: str, model_path: str, contamination: float) -> None:
    """Huấn luyện Isolation Forest và lưu model + scaler"""
    df = pd.read_csv(file_path)
    # Chọn các đặc trưng số.  Nếu thiếu sẽ thay thế bằng 0.
    features = df[['status', 'response_time', 'bytes']].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)
    joblib.dump({'model': clf, 'scaler': scaler}, model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument('--file', required=True, help='Path to CSV file containing logs')
    parser.add_argument('--model-file', default='model.pkl', help='Output file for trained model')
    parser.add_argument('--contamination', type=float, default=0.01, help='Proportion of anomalies in the data')
    args = parser.parse_args()
    train_model(args.file, args.model_file, args.contamination)


if __name__ == '__main__':
    main()