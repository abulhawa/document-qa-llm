import sqlite3
import json
import logging
from config import DATABASE_PATH


def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_tables():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                checksum TEXT,
                last_indexed TEXT
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                checksum TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
        """
                )
        conn.commit()


def is_file_up_to_date(path: str, checksum: str) -> bool:
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT checksum FROM files WHERE path = ?", (path,))
        row = c.fetchone()
        return row is not None and row[0] == checksum


def upsert_file_metadata(path: str, checksum: str, last_indexed: str):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO files (path, checksum, last_indexed)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                checksum=excluded.checksum,
                last_indexed=excluded.last_indexed
        """,
            (path, checksum, last_indexed),
        )
        conn.commit()

       
def insert_chunks(path: str, checksum: str, chunks):
    with get_db_connection() as conn:
        c = conn.cursor()
        records = [
            (path, checksum, i, content, json.dumps(embed))
            for i, (content, embed) in enumerate(chunks)
        ]
        c.executemany("""
            INSERT INTO chunks (file_path, checksum, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        logging.info(f"✅ Inserted {len(chunks)} chunks for {path}")


def fetch_all_chunks():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT checksum, chunk_index, embedding FROM chunks")
        return c.fetchall()


def insert_faiss_mapping(faiss_id: int, checksum: str, chunk_index: int):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO faiss_map (faiss_id, checksum, chunk_index)
            VALUES (?, ?, ?)
        """,
            (faiss_id, checksum, chunk_index),
        )
        conn.commit()


def is_checksum_in_faiss(checksum: str) -> bool:
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM faiss_map WHERE checksum = ?", (checksum,))
        return c.fetchone()[0] > 0


def get_indexed_chunk_count():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_file_record_by_path(path: str):
    ## TODO this only checks the path then compares the checksum... however, i think we need only check if the checksum exists regardless of the path
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT path, checksum, last_indexed FROM files WHERE path = ?", (path,)
        )
        row = c.fetchone()
        if row:
            return {"path": row[0], "checksum": row[1], "last_indexed": row[2]}
        return None


def get_all_chunks_with_embeddings():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            """            
            SELECT id, file_path, checksum, chunk_index, content, embedding
            FROM chunks
        """
        )
        rows = c.fetchall()
        chunks = []
        for row in rows:
            chunks.append({
                "id": row[0],
                "file_path": row[1],
                "checksum": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "embedding": json.loads(row[5])
            })
        return chunks


def get_chunk_by_ids(ids: list[int]):
    if not ids:
        return []

    placeholders = ",".join("?" for _ in ids)

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(f"""
            SELECT id, file_path, checksum, chunk_index, content, embedding
            FROM chunks
            WHERE id IN ({placeholders})
        """, ids)
        rows = c.fetchall()
        return [
            {
                "id": row[0],
                "file_path": row[1],
                "checksum": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "embedding": json.loads(row[5])
            }
            for row in rows
        ]