# IMDELD 변압기 CSV 재생용
# 실제 공장 1초 데이터에서 전력(kW)과 전압 흔들림 비율을 꺼내 시뮬레이터에 공급

import numpy as np
import pandas as pd


class Replay:

    # CSV 로드 + 규모 맞추기
    # 원본 공장은 수십 kW 규모라, 가동 중 상위 5% 전력이 p95_kw 가 되도록 배율을 곱함
    # 전압은 고압측(약 6.8kV) 값이라 중앙값 대비 비율만 사용
    # replay_from: 이 날짜 이후만 재생 (학습에 안 쓴 test 기간). 배율은 전체 데이터 기준 유지
    def __init__(self, path, p95_kw, start_hour, start_minute, replay_from=None):
        print(f"재생 데이터 로드 중: {path}")
        df = pd.read_csv(path, usecols=["timestamp", "active_power", "voltage"],
                         dtype={"active_power": "float32", "voltage": "float32"})

        # "2017-12-11 18:43:52-02" -> 앞 19자리만 사용 (시간대 표기 제거)
        ts = pd.to_datetime(df["timestamp"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")

        kw = (df["active_power"].clip(lower=0) / 1000).to_numpy()
        running = kw[kw > kw.max() * 0.05]          # 공장 정지 구간 제외
        self.scale = p95_kw / np.percentile(running, 95)
        self.kw = (kw * self.scale).astype("float32")

        v = df["voltage"].to_numpy()
        v_med = np.median(v[v > 0])
        self.v_ratio = np.where(v > 0, v / v_med, 1.0).astype("float32")

        self.ts = ts.to_numpy()
        self.first = self.find_first(ts, replay_from)
        self.i = self.find_start(ts, start_hour, start_minute)
        del df

        print(f"재생 준비 완료: {len(self.kw):,}행 · 배율 x{self.scale:.1f} · "
              f"범위 {pd.Timestamp(self.ts[self.first]):%Y-%m-%d}~{pd.Timestamp(self.ts[-1]):%Y-%m-%d} · "
              f"시작 {pd.Timestamp(self.ts[self.i]):%Y-%m-%d %H:%M:%S}")

    # 재생 범위의 첫 행 위치. replay_from 이 없거나 데이터보다 늦으면 처음부터
    def find_first(self, ts, replay_from):
        if not replay_from:
            return 0
        idx = np.flatnonzero(ts >= pd.Timestamp(replay_from))
        if len(idx) == 0:
            print(f"REPLAY_FROM={replay_from} 이후 데이터 없음 -> 전체 재생")
            return 0
        return int(idx[0])

    # 재생 범위 안에서 지금 시각(시:분)과 같은 시각의 첫 행부터 재생
    def find_start(self, ts, hour, minute):
        tod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        idx = np.flatnonzero(tod == hour * 60 + minute)
        idx = idx[idx >= self.first]
        if len(idx) == 0:
            return self.first
        return int(idx[0])

    # 1초분 꺼내기. 끝까지 가면 재생 범위 처음(first)으로 다시
    def next(self):
        kw = float(self.kw[self.i])
        v_ratio = float(self.v_ratio[self.i])
        src_ts = pd.Timestamp(self.ts[self.i])

        self.i += 1
        if self.i >= len(self.kw):
            self.i = self.first
        return kw, v_ratio, src_ts