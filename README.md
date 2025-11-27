# Dự án Phát hiện Bất thường Log Web với ELK Stack

## Giới thiệu

Đây là dự án mẫu giúp bạn triển khai hệ thống phát hiện bất thường trên log web server bằng cách kết hợp **ELK Stack** (Elasticsearch, Logstash, Kibana) và mô hình học máy. Dự án bao gồm các file cấu hình Docker, pipeline của Logstash, dữ liệu mẫu, mã Python để huấn luyện và đánh giá mô hình, cũng như hướng dẫn từng bước để vận hành.

Tài liệu này giả định bạn đã đọc báo cáo tổng quan đính kèm và hiểu cách thức ELK hoạt động. Một số nguồn tham khảo quan trọng:

- Bài viết về Logstash trong series “Làm chủ ELK Stack” hướng dẫn cấu hình pipeline input–filter–output【108337509752359†L120-L139】.
- Bài viết “Mastering Anomaly Detection with the ELK Stack” trình bày các bước chuẩn bị dữ liệu và tạo job anomaly detection trong Kibana【147792481961476†L73-L90】.
- Dataset CIC‑IDS 2017/2018 cung cấp log thực tế với nhiều loại tấn công khác nhau【586822576890327†L132-L137】【139013984303761†L141-L147】.

## Cấu trúc thư mục

```
project/
├── docker-compose.yml     # cấu hình khởi động Elasticsearch, Kibana và Logstash
├── logstash.conf          # pipeline Logstash để đọc file CSV và gửi vào Elasticsearch
├── data/
│   └── sample_logs.csv    # dữ liệu log mẫu (10 dòng) dùng để thử nghiệm
├── requirements.txt       # danh sách thư viện Python cần cài
├── train_model.py         # script huấn luyện mô hình Isolation Forest
├── ingest_data.py         # script đọc file CSV và index vào Elasticsearch
├── detect_anomalies.py    # script đọc file CSV, tính điểm bất thường bằng mô hình đã huấn luyện
└── README.md              # hướng dẫn sử dụng (file bạn đang đọc)
```

Bạn có thể thay thế `data/sample_logs.csv` bằng dataset thực tế (ví dụ file CSV từ CIC‑IDS2017) bằng cách đặt file trong thư mục `data` và cập nhật các tham số dòng lệnh cho script.

## Yêu cầu

1. **Docker**: Để chạy Elasticsearch, Logstash và Kibana trong container.
2. **Python 3.8+** và `pip`: Để chạy các script huấn luyện và đánh giá mô hình.
3. **File dữ liệu**: Bạn nên tải dataset từ UNB (CIC‑IDS 2017/2018 hoặc IoT‑DIAD 2024) theo hướng dẫn ở báo cáo. Các script chấp nhận file CSV có header `timestamp,src_ip,dst_ip,method,url,status,response_time,bytes`.

## Các bước triển khai

### 1. Clone dự án và chuẩn bị môi trường

Sao chép thư mục `project` (đã kèm theo trong file zip) vào máy của bạn. Cài đặt các thư viện Python bằng pip:

```bash
cd project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Khởi động ELK Stack bằng Docker Compose

Dùng `docker-compose` để chạy các container Elasticsearch, Kibana và Logstash:

```bash
docker-compose up -d
```

Docker sẽ tự động kéo hình ảnh phù hợp (phiên bản 8.5.3) và khởi động các dịch vụ. Bạn có thể kiểm tra trạng thái bằng `docker-compose ps`.

Đợi khoảng 1 – 2 phút để Elasticsearch và Kibana sẵn sàng. Kibana sẽ chạy ở `http://localhost:5601` và Elasticsearch ở `http://localhost:9200`.

### 3. Ingest dữ liệu vào Elasticsearch

Sau khi các container chạy, bạn có thể nạp dữ liệu CSV vào chỉ mục `web-logs` bằng script Python. Script này đọc file CSV và gửi từng dòng vào Elasticsearch thông qua API:

```bash
python ingest_data.py --file data/sample_logs.csv --index web-logs
```

Nếu sử dụng dataset lớn, hãy điều chỉnh tham số `--batch-size` (mặc định 5000) để tối ưu hiệu suất. Bạn cũng có thể thay đổi URL của Elasticsearch thông qua `--es-host` (mặc định `http://localhost:9200`).

### 4. Huấn luyện mô hình Isolation Forest

Script `train_model.py` huấn luyện mô hình phát hiện bất thường trên dữ liệu. Mô hình sử dụng trường `status`, `response_time` và `bytes` làm đặc trưng, sau đó chuẩn hóa và huấn luyện Isolation Forest:

```bash
python train_model.py --file data/sample_logs.csv --model-file model.pkl
```

Bạn có thể chỉ định mức độ nhiễm bẩn (contamination) với tham số `--contamination` (mặc định 0.01). Kết quả huấn luyện được lưu trong file `model.pkl`.

### 5. Tính điểm bất thường trên dữ liệu

Sau khi có mô hình, sử dụng script `detect_anomalies.py` để tính điểm bất thường cho từng dòng log. Script sẽ sinh file `anomaly_scores.csv` bao gồm cột `anomaly_score` và nhãn (`anomaly_label`: -1 = bất thường, 1 = bình thường):

```bash
python detect_anomalies.py --file data/sample_logs.csv --model-file model.pkl --output anomaly_scores.csv
```

Bạn có thể dùng file này để cập nhật lại index trong Elasticsearch (ví dụ qua Bulk API) hoặc trực quan hoá kết quả bằng Pandas.

### 6. Trực quan hóa và tạo cảnh báo trong Kibana

1. Mở `http://localhost:5601` trong trình duyệt.
2. Vào **Stack Management → Kibana → Data Views** và tạo Data View với tên `web-logs*` chọn trường thời gian là `@timestamp`.
3. Vào **Discover** để xem dữ liệu log đã được index và thực hiện truy vấn.
4. (Tuỳ chọn) Thiết lập **Machine Learning → Anomaly Detection** để tạo job phát hiện bất thường ngay trong Kibana theo hướng dẫn của Elastic【147792481961476†L73-L90】.
5. Tạo dashboard hiển thị số request, status code, và biểu đồ `anomaly_score`. Bạn có thể sử dụng tính năng Alerting của Kibana để gửi thông báo khi có bất thường.

## Nội dung các file Python

### train_model.py

```python
#!/usr/bin/env python
"""
Huấn luyện mô hình Isolation Forest trên dữ liệu log.
Chọn các trường số: status, response_time, bytes để làm đặc trưng.
Model và scaler được lưu bằng joblib.
"""
import argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(file_path: str, model_path: str, contamination: float):
    df = pd.read_csv(file_path)
    features = df[['status', 'response_time', 'bytes']].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)
    joblib.dump({'model': clf, 'scaler': scaler}, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--model-file', default='model.pkl', help='Output model file path')
    parser.add_argument('--contamination', type=float, default=0.01, help='Expected proportion of anomalies')
    args = parser.parse_args()
    train_model(args.file, args.model_file, args.contamination)
```

### ingest_data.py

```python
#!/usr/bin/env python
"""
Script để đọc file CSV và index dữ liệu vào Elasticsearch.
Bạn có thể thay đổi batch_size để tối ưu hiệu suất khi xử lý file lớn.
"""
import argparse
import pandas as pd
from elasticsearch import Elasticsearch, helpers

def ingest(file_path: str, index_name: str, es_host: str = 'http://localhost:9200', batch_size: int = 5000):
    es = Elasticsearch(es_host)
    df = pd.read_csv(file_path)
    actions = []
    for i, row in df.iterrows():
        doc = {
            '_op_type': 'index',
            '_index': index_name,
            '_source': {
                'timestamp': row['timestamp'],
                'src_ip': row['src_ip'],
                'dst_ip': row['dst_ip'],
                'method': row['method'],
                'url': row['url'],
                'status': int(row['status']),
                'response_time': float(row['response_time']),
                'bytes': int(row['bytes'])
            }
        }
        actions.append(doc)
        if len(actions) >= batch_size:
            helpers.bulk(es, actions)
            actions = []
    if actions:
        helpers.bulk(es, actions)
    print(f"Indexed {len(df)} documents into index '{index_name}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest CSV data into Elasticsearch')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--index', default='web-logs', help='Name of the Elasticsearch index')
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host URL')
    parser.add_argument('--batch-size', type=int, default=5000, help='Number of documents per batch')
    args = parser.parse_args()
    ingest(args.file, args.index, args.es_host, args.batch_size)
```

### detect_anomalies.py

```python
#!/usr/bin/env python
"""
Tính điểm bất thường và nhãn dựa trên mô hình Isolation Forest đã huấn luyện.
Output: file CSV chứa anomaly_score và anomaly_label.
"""
import argparse
import pandas as pd
import joblib

def detect(file_path: str, model_path: str, output_path: str):
    model_data = joblib.load(model_path)
    clf = model_data['model']
    scaler = model_data['scaler']
    df = pd.read_csv(file_path)
    X = scaler.transform(df[['status', 'response_time', 'bytes']].fillna(0))
    scores = clf.decision_function(X)
    labels = clf.predict(X)
    result_df = df.copy()
    result_df['anomaly_score'] = scores
    result_df['anomaly_label'] = labels
    result_df.to_csv(output_path, index=False)
    print(f"Saved anomaly scores to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect anomalies using a trained model')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--model-file', required=True, help='Path to trained model (joblib file)')
    parser.add_argument('--output', default='anomaly_scores.csv', help='Output CSV file path')
    args = parser.parse_args()
    detect(args.file, args.model_file, args.output)
```

## Ghi chú

- **Cấu hình Logstash**: Pipeline `logstash.conf` được thiết kế để đọc file `sample_logs.csv` trong thư mục `data`. Nếu bạn muốn Logstash tự động ingest dataset khác, hãy đặt tên file tương ứng và cập nhật đường dẫn trong `logstash.conf`.
- **Sử dụng dataset thực tế**: Các dataset như CIC‑IDS2017/2018 và IoT‑DIAD 2024 chứa hàng triệu dòng. Bạn nên kiểm tra dung lượng trước khi ingest và có thể sử dụng Filebeat để gửi log trực tiếp từ máy tính tới Logstash, thay vì đọc file lớn trong container.
- **ML trong Kibana**: Elastic cung cấp tính năng Anomaly Detection tích hợp. Bạn có thể tạo job ngay trong Kibana để theo dõi các trường số như `response_time` hoặc `bytes` theo thời gian【147792481961476†L73-L90】.
- **Tối ưu hoá**: Khi chạy trên dataset lớn, hãy điều chỉnh `ES_JAVA_OPTS` trong `docker-compose.yml` và sử dụng cấu hình index lifecycle để xoá dữ liệu cũ.

Hy vọng dự án mẫu này giúp bạn triển khai giải pháp phát hiện bất thường trên log web server với ELK Stack và machine learning. Nếu có câu hỏi hoặc cần hỗ trợ, vui lòng tham khảo lại báo cáo tổng quan hoặc tài liệu chính thức của Elastic.