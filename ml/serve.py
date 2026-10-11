# 서빙: model.pkl 로드 -> 30초마다 예측 -> 구간 끝나면 평가 -> 하루치 쌓이면 재학습
#
# 사용법 (ml 폴더에서, simulator · controller · server 가 켜진 상태):
#   python serve.py
#
# 주기
#   30초마다       : 진행 중인 15분 구간의 평균 예측 -> ml.forecast 저장
#   구간 끝날 때마다 : 실제 평균과 비교 (계산식 오차 vs ML 오차) -> ml.window_eval 저장, 화면 출력
#   새 구간 96개마다 : train.py --include-db 실행 -> 더 좋으면 model.pkl 교체 -> 다시 로드
#
# 그림자 운영: ML 예측은 기록만 하고, 실제 제어는 controller 의 기존 계산식이 계속 담당

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import features
from models import predict_bundle

PREDICT_SEC = 30
RETRAIN_WINDOWS = 96          # 15분 x 96 = 하루
RETRAIN_TRIALS = 5            # 재학습은 짧게
MODEL_PATH = "models/model.pkl"

load_dotenv()
DB_URL = os.getenv("DB_URL")


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def create_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml.forecast (
              ts TIMESTAMP PRIMARY KEY,
              window_start TIMESTAMP,
              elapsed_min REAL,
              formula_kw REAL,
              ml_kw REAL,
              run_id TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml.window_eval (
              window_start TIMESTAMP PRIMARY KEY,
              actual_kw REAL,
              formula_mae REAL,
              ml_mae REAL,
              n_points INT,
              run_id TEXT
            )"""))


def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    print(f"모델 로드: {'+'.join(bundle['members'])} (run {bundle['run_id']})")
    return bundle


# 진행 중인 구간 1회 예측
def predict_once(engine, bundle):
    t = now_kst()
    ws = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)
    k = int((t - ws).total_seconds() // features.STEP_SEC)
    if k < features.FIRST_POINT:
        return                         # 구간 시작 2분 이내는 예측 안 함

    # 최근 30분 데이터 (최근 10분 시퀀스 + 이번 구간)
    since = t - timedelta(minutes=30)
    steps = features.load_db_power(DB_URL, since=since)
    if len(steps) == 0 or ws not in steps.index:
        print(f"[{t:%H:%M:%S}] 데이터 부족 -> 건너뜀")
        return

    values = steps.to_numpy(dtype="float64")
    start_pos = steps.index.get_loc(ws)
    point = features.make_point(values, start_pos, k, ws)
    if point is None:
        print(f"[{t:%H:%M:%S}] 빈 구간 있음 -> 건너뜀")
        return

    tab, seq = point
    ml_kw = float(predict_bundle(bundle, pd.DataFrame([tab]), seq[None, :])[0])
    formula_kw = tab["proj_base"]

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ml.forecast VALUES (:ts, :ws, :em, :f, :m, :run)
            ON CONFLICT (ts) DO NOTHING"""),
            {"ts": t, "ws": ws, "em": k * features.STEP_SEC / 60,
             "f": formula_kw, "m": ml_kw, "run": bundle["run_id"]})

    print(f"[{t:%H:%M:%S}] 구간 {ws:%H:%M} | 계산식 {formula_kw:7.0f} kW | ML {ml_kw:7.0f} kW")


# 끝난 구간 평가 (ops.demand_15min 의 확정 평균과 비교). 새로 평가한 구간 수 반환
def evaluate_closed(engine, bundle):
    sql = text("""
        SELECT f.window_start,
               d.avg_kw AS actual_kw,
               avg(abs(f.formula_kw - d.avg_kw)) AS formula_mae,
               avg(abs(f.ml_kw - d.avg_kw)) AS ml_mae,
               count(*) AS n_points
        FROM ml.forecast f
        JOIN ops.demand_15min d ON d.window_start = f.window_start AND d.final
        WHERE f.window_start NOT IN (SELECT window_start FROM ml.window_eval)
        GROUP BY f.window_start, d.avg_kw
        ORDER BY f.window_start""")

    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
        for r in rows:
            conn.execute(text("""
                INSERT INTO ml.window_eval VALUES (:ws, :a, :fm, :mm, :n, :run)
                ON CONFLICT (window_start) DO NOTHING"""),
                {"ws": r["window_start"], "a": r["actual_kw"], "fm": r["formula_mae"],
                 "mm": r["ml_mae"], "n": r["n_points"], "run": bundle["run_id"]})

            if r["ml_mae"] < r["formula_mae"]:
                winner = "ML 승"
            else:
                winner = "계산식 승"
            print(f"[구간 평가] {r['window_start']:%H:%M} 실제 {r['actual_kw']:,.0f} kW | "
                  f"계산식 오차 {r['formula_mae']:.0f} kW | ML 오차 {r['ml_mae']:.0f} kW | {winner}")

    return len(rows)


def retrain():
    print("=== 재학습 시작 (끝날 때까지 예측 잠시 멈춤) ===")
    result = subprocess.run([sys.executable, "train.py", "--include-db", "--trials", str(RETRAIN_TRIALS)])
    print("=== 재학습 끝 ===")
    return result.returncode == 0


def main():
    if not os.path.exists(MODEL_PATH):
        print("models/model.pkl 이 없습니다. 먼저 python train.py 를 실행하세요.")
        return

    engine = create_engine(DB_URL, pool_pre_ping=True)
    create_tables(engine)
    bundle = load_bundle()
    new_windows = 0

    while True:
        t0 = time.time()

        try:
            predict_once(engine, bundle)
            new_windows = new_windows + evaluate_closed(engine, bundle)
        except Exception as e:
            print("오류:", e)

        if new_windows >= RETRAIN_WINDOWS:
            if retrain():
                bundle = load_bundle()
            new_windows = 0

        elapsed = time.time() - t0
        time.sleep(max(0.0, PREDICT_SEC - elapsed))


if __name__ == "__main__":
    main()
