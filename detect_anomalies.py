#!/usr/bin/env python
"""
Tính điểm bất thường bằng mô hình Isolation Forest đã huấn luyện.

Script này đọc file CSV, nạp model đã huấn luyện (chứa Isolation Forest và
StandardScaler), áp dụng scaler lên các đặc trưng số (status, response_time,
bytes) rồi tính `decision_function` để lấy anomaly_score và nhãn (-1
cho bất thường, 1 cho bình thường).  Kết quả được lưu vào file CSV
mới với hai cột `anomaly_score` và `anomaly_label`.

Ví dụ:
    python detect_anomalies.py --file data/sample_logs.csv --model-file model.pkl --output anomaly_scores.csv
"""

import argparse
import pandas as pd
import joblib


def detect(file_path: str, model_path: str, output_path: str) -> None:
    model_data = joblib.load(model_path)
    clf = model_data['model']
    scaler = model_data['scaler']
    df = pd.read_csv(file_path)
    features = df[['status', 'response_time', 'bytes']].fillna(0)
    X = scaler.transform(features)
    scores = clf.decision_function(X)
    labels = clf.predict(X)
    df['anomaly_score'] = scores
    df['anomaly_label'] = labels
    df.to_csv(output_path, index=False)
    print(f"Saved anomaly scores to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect anomalies in log data')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--model-file', required=True, help='Path to trained model (joblib)')
    parser.add_argument('--output', default='anomaly_scores.csv', help='Output CSV file')
    args = parser.parse_args()
    detect(args.file, args.model_file, args.output)


if __name__ == '__main__':
    main()