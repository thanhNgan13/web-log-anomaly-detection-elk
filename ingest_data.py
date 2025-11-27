#!/usr/bin/env python
"""
Script để đọc file CSV và index dữ liệu vào Elasticsearch.

Kịch bản này sử dụng thư viện `elasticsearch` và `helpers.bulk` để nạp dữ
liệu một cách hiệu quả.  Đối với file lớn, nên điều chỉnh `batch_size` để
tránh tốn bộ nhớ.  Bạn cũng có thể chỉ định tên index và host qua tham số.

Ví dụ:
    python ingest_data.py --file data/sample_logs.csv --index web-logs
"""

import argparse
import pandas as pd
from elasticsearch import Elasticsearch, helpers


def ingest(file_path: str, index_name: str, es_host: str = 'http://localhost:9200', batch_size: int = 5000) -> None:
    """Đọc CSV và index vào Elasticsearch"""
    es = Elasticsearch(es_host)
    df = pd.read_csv(file_path)
    actions = []
    count = 0
    for _, row in df.iterrows():
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
            count += len(actions)
            actions = []
    if actions:
        helpers.bulk(es, actions)
        count += len(actions)
    print(f"Indexed {count} documents into index '{index_name}'")


def main():
    parser = argparse.ArgumentParser(description='Ingest CSV data into Elasticsearch')
    parser.add_argument('--file', required=True, help='Path to CSV file')
    parser.add_argument('--index', default='web-logs', help='Elasticsearch index name')
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host URL')
    parser.add_argument('--batch-size', type=int, default=5000, help='Number of documents per bulk request')
    args = parser.parse_args()
    ingest(args.file, args.index, args.es_host, args.batch_size)


if __name__ == '__main__':
    main()