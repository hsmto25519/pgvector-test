import json
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

# 1. 事前に用意した3,000件の日本語文章データを読み込む
with open("sentences_3000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Embeddingモデルで3,000件の文章をベクトル化（E5モデルの仕様に合わせて "passage: " を付与）
model = SentenceTransformer(MODEL_NAME)
passages = ["passage: " + item["sentence"] for item in data]
embeddings = model.encode(passages, batch_size=128, show_progress_bar=True)

# 3. PostgreSQLへ接続してテーブル作成・データ投入
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="vector_db",
    user="postgres",
    password="secret0001",
)
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cursor.execute("DROP TABLE IF EXISTS items;")
cursor.execute("""
    CREATE TABLE items (
        id SERIAL PRIMARY KEY,
        category TEXT,
        sentence TEXT,
        embedding VECTOR(384)
    );
""")

rows = []
for item, emb in zip(data, embeddings):
    vec_str = "[" + ",".join(f"{float(v):.6f}" for v in emb) + "]"
    rows.append((item["category"], item["sentence"], vec_str))

execute_values(
    cursor,
    "INSERT INTO items (category, sentence, embedding) VALUES %s",
    rows,
)
cursor.execute("ANALYZE items;")

# 4. 検索用のクエリ文とベクトルを別テーブルに1件保存（検索クエリには "query: " を付与）
query_text = "PostgreSQLのインデックスを使ってベクトル検索を高速化する方法"
query_emb = model.encode("query: " + query_text)
query_vec = "[" + ",".join(f"{float(v):.6f}" for v in query_emb) + "]"

cursor.execute("DROP TABLE IF EXISTS query_target;")
cursor.execute("""
    CREATE TABLE query_target (
        q_text TEXT,
        q_vec  VECTOR(384)
    );
""")
cursor.execute(
    "INSERT INTO query_target VALUES (%s, %s::vector);",
    (query_text, query_vec),
)
cursor.execute("ANALYZE query_target;")

cursor.close()
conn.close()
print("done")
