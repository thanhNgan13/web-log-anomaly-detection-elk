#!/usr/bin/env python
"""
Tính điểm bất thường bằng mô hình Isolation Forest đã huấn luyện.

Script này đọc file CSV, nạp model đã huấn luyện (chứa Isolation Forest và
StandardScaler), áp dụng scaler lên các đặc trưng số (status, response_time,
bytes) rồi tính `decision_function` để lấy anomaly_score và nhãn (-1
cho bất thường, 1 cho bình thường).  Kết quả được lưu vào file CSV
mới với hai cột `anomaly_score` và `anomaly_label`.

Ví dụ:
    python detect_anomalies.py --file data/sample_logs.csv --model-file model.pkl --output anomaly_scores.csv --index-results
"""

import argparse
import pandas as pd
import joblib
from elasticsearch import Elasticsearch, helpers


def detect(file_path: str, model_path: str, output_path: str, index_results: bool, es_index: str, es_host: str) -> None:
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

    if index_results:
        print(f"Indexing results to Elasticsearch index '{es_index}'...")
        es = Elasticsearch(es_host)

        # Create index with mapping if it doesn't exist
        if not es.indices.exists(index=es_index):
            mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {
                            "type": "date",
                            "format": "yyyy-MM-dd HH:mm:ss"
                        },
                        "src_ip": {"type": "ip"},
                        "dst_ip": {"type": "ip"},
                        "anomaly_score": {"type": "float"},
                        "anomaly_label": {"type": "integer"},
                        "status": {"type": "integer"},
                        "response_time": {"type": "float"},
                        "bytes": {"type": "long"}
                    }
                }
            }
            es.indices.create(index=es_index, body=mapping)
            print(f"Created index '{es_index}' with mapping.")

        actions = []
        for _, row in df.iterrows():
            doc = row.to_dict()
            action = {
                '_index': es_index,
                '_source': doc
            }
            actions.append(action)
        
        if actions:
            helpers.bulk(es, actions)
            print(f"Indexed {len(actions)} documents into '{es_index}'")


def main():
    parser = argparse.ArgumentParser(description='Detect anomalies in log data')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--model-file', required=True, help='Path to trained model (joblib)')
    parser.add_argument('--output', default='anomaly_scores.csv', help='Output CSV file')
    parser.add_argument('--index-results', action='store_true', help='Index results to Elasticsearch')
    parser.add_argument('--es-index', default='web-logs-anomalies', help='Elasticsearch index name for results')
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host URL')
    args = parser.parse_args()
    detect(args.file, args.model_file, args.output, args.index_results, args.es_index, args.es_host)


if __name__ == '__main__':
    main()