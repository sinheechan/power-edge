import sqlite3

# MQTT 끊겼을 때 못 보낸 메시지 임시 저장용 (SQLite)
# 파일로 저장해서 프로그램 재시작해도 남아 있음

class Buffer:

    # DB 연결 + 테이블 생성
    # check_same_thread=False -> MQTT 스레드에서도 같이 쓰기 위해
    # id는 자동 증가라서 쌓인 순서대로 다시 보낼 수 있음
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread = False)
        self.db.execute("CREATE TABLE IF NOT EXISTS buf "
                        "(id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, payload TEXT)")
        self.db.commit()

    # 못 보낸 메시지 저장
    def put(self, topic, payload):
        self.db.execute("INSERT INTO buf (topic, payload) VALUES (?, ?)", (topic, payload))
        self.db.commit()

    # 오래된 것부터 n개 가져오기 (삭제는 안 함)
    def take(self, n=200):
        return self.db.execute("SELECT id, topic, payload FROM buf ORDER BY id LIMIT ?", (n,)).fetchall()

    # 재전송 끝난 것만 삭제
    def delete(self, ids):
        self.db.executemany("DELETE FROM buf WHERE id = ?", [(i,) for i in ids])
        self.db.commit()

    # 남아 있는 개수
    def size(self):
        return self.db.execute("SELECT count(*) FROM buf").fetchone()[0]