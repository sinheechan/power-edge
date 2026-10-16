# 특징 만들기 (eda.ipynb, train.py, serve.py 가 같이 씀)
#
# 1초 전력 -> 10초 평균(1칸) -> 15분 구간(90칸) 안의 예측 시점마다
# 입력(요약 특징 + 최근 10분 시퀀스)과 정답(구간 실제 평균)을 만듦
# 학습과 서빙이 같은 make_point 를 써야 결과가 어긋나지 않음

import os

import numpy as np
import pandas as pd

ML_DIR = os.path.dirname(os.path.abspath(__file__))   # 어디서 실행해도 ml 폴더 기준으로 경로 잡기
CSV_PATH = os.path.join(ML_DIR, "..", "data", "mvlv-transformer.csv")
P95_KW = 1400        # simulator 와 같은 규모 (가동 중 상위 5% 전력)

STEP_SEC = 10        # 10초 평균 = 1칸
WINDOW = 90          # 15분 = 90칸
HIST = 60            # 최근 10분 = 60칸
FIRST = 12           # 구간 시작 2분 후부터 예측 (controller 판단 보류와 같음)
LAST = 84            # 구간 끝 1분 전까지
EVERY = 6            # 학습 샘플 간격 6칸 = 1분
NORM = 1000.0        # 딥러닝 입력 크기 맞추기용

# 트리, 선형 모델 입력
TAB_COLS = [
    "elapsed_ratio", "used_kwh", "kw_now", "proj_base", "proj_trend",
    "mean_1m", "mean_5m", "mean_10m", "std_5m", "slope_5m", "max_10m", "min_10m",
    "hour_sin", "hour_cos", "dow",
]

# 딥러닝 모델이 시퀀스와 같이 쓰는 입력 (TAB_COLS 중 일부)
EXTRA_COLS = ["elapsed_ratio", "proj_base", "kw_now", "hour_sin", "hour_cos"]


# 1초 데이터 -> 10초 평균. 빠진 시간은 NaN
def to_steps(s):
    s = s.sort_index()
    s = s[~s.index.duplicated()]
    return s.resample(f"{STEP_SEC}s").mean()


# CSV -> 10초 평균 kW (simulator 재생과 같은 배율)
def load_csv():
    df = pd.read_csv(CSV_PATH, usecols=["timestamp", "active_power"], dtype={"active_power": "float32"})

    # "2017-12-11 18:43:52-02" -> 앞 19글자만 (시간대 표기 버림)
    ts = pd.to_datetime(df["timestamp"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    kw = (df["active_power"].clip(lower=0) / 1000).to_numpy()

    # 공장 쉬는 시간 빼고 상위 5% 가 P95_KW 가 되도록 배율
    running = kw[kw > kw.max() * 0.05]
    scale = P95_KW / np.percentile(running, 95)

    return to_steps(pd.Series(kw * scale, index=ts)), scale


# DB(raw.meter_reading) -> 10초 평균 kW. since 가 있으면 그 시각 이후만
def load_db(db_url, since=None):
    from sqlalchemy import create_engine, text

    sql = "SELECT ts, kw FROM raw.meter_reading"
    params = {}
    if since is not None:
        sql += " WHERE ts >= :since"
        params["since"] = since
    sql += " ORDER BY ts"

    with create_engine(db_url).connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return pd.Series(dtype="float64")
    s = pd.Series(df["kw"].to_numpy(dtype="float64"), index=pd.to_datetime(df["ts"]))
    return to_steps(s)


# 한 예측 시점의 특징
#   values: 10초 평균 배열, start: 구간 시작 칸, k: 구간 시작 후 지난 칸 수
#   반환: (요약 특징 dict, 최근 10분 시퀀스). 데이터가 비어 있으면 None
def make_point(values, start, k, ws):
    now = start + k
    if now - HIST < 0 or now > len(values):
        return None

    hist = values[now - HIST:now]
    inside = values[start:now]
    if len(hist) < HIST or np.isnan(hist).any() or np.isnan(inside).any():
        return None

    kw_now = float(hist[-1])
    elapsed_h = k * STEP_SEC / 3600
    remain_h = 0.25 - elapsed_h
    used_kwh = float(inside.sum()) * STEP_SEC / 3600

    # 기준 1: 기존 계산식 (지금 kW 가 끝까지 간다고 가정)
    proj_base = (used_kwh + kw_now * remain_h) / 0.25

    # 기준 2: 최근 5분 기울기대로 계속 오르내린다고 가정
    last5 = hist[-30:]
    slope = float(np.polyfit(np.arange(len(last5)), last5, 1)[0])   # 1칸당 kW 변화
    slope_h = slope * 3600 / STEP_SEC                                # 1시간당 kW 변화
    future_kwh = kw_now * remain_h + 0.5 * slope_h * remain_h ** 2
    if future_kwh < 0:
        future_kwh = 0.0
    proj_trend = (used_kwh + future_kwh) / 0.25

    # 서빙에선 ws 가 datetime 으로 와서 Timestamp 로 바꿈 (datetime 엔 dayofweek 없음)
    now_time = pd.Timestamp(ws) + pd.Timedelta(seconds=k * STEP_SEC)
    hour = now_time.hour + now_time.minute / 60

    tab = {
        "elapsed_ratio": k / WINDOW,
        "used_kwh": used_kwh,
        "kw_now": kw_now,
        "proj_base": proj_base,
        "proj_trend": proj_trend,
        "mean_1m": float(hist[-6:].mean()),
        "mean_5m": float(hist[-30:].mean()),
        "mean_10m": float(hist.mean()),
        "std_5m": float(hist[-30:].std()),
        "slope_5m": slope,
        "max_10m": float(hist.max()),
        "min_10m": float(hist.min()),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow": now_time.dayofweek,
    }
    seq = (hist / NORM).astype("float32")
    return tab, seq


# 10초 평균 시계열 -> 학습 샘플 전체
#   반환: df (요약 특징 + window_start, date, elapsed_min, actual_kw), X_seq (샘플 수 x 60)
def make_samples(steps):
    values = steps.to_numpy(dtype="float64")
    rows = []
    seqs = []

    # 데이터 범위 안의 15분 구간들
    first = steps.index[0].ceil("15min")
    last = steps.index[-1].floor("15min") - pd.Timedelta(minutes=15)

    for ws in pd.date_range(first, last, freq="15min"):
        start = steps.index.get_loc(ws)
        window = values[start:start + WINDOW]

        # 구간 데이터가 다 있어야 정답(실제 평균)을 앎
        if len(window) < WINDOW or np.isnan(window).any():
            continue
        actual = float(window.mean())

        for k in range(FIRST, LAST + 1, EVERY):
            point = make_point(values, start, k, ws)
            if point is None:
                continue

            tab, seq = point
            tab["window_start"] = ws
            tab["date"] = ws.date()
            tab["elapsed_min"] = k * STEP_SEC / 60
            tab["actual_kw"] = actual
            rows.append(tab)
            seqs.append(seq)

    df = pd.DataFrame(rows)
    if len(seqs) > 0:
        X_seq = np.stack(seqs)
    else:
        X_seq = np.zeros((0, HIST), dtype="float32")
    return df, X_seq


# 날짜 기준 분할: 앞 80% 학습 / 다음 10% 검증 / 마지막 10% 시험
# 랜덤으로 섞으면 같은 날이 학습과 시험에 같이 들어가서 성능이 부풀려짐
def split_by_date(df):
    dates = sorted(df["date"].unique())
    n_train = int(len(dates) * 0.8)
    n_valid = int(len(dates) * 0.1)

    train_dates = set(dates[:n_train])
    valid_dates = set(dates[n_train:n_train + n_valid])

    split = []
    for d in df["date"]:
        if d in train_dates:
            split.append("train")
        elif d in valid_dates:
            split.append("valid")
        else:
            split.append("test")
    return np.array(split)
