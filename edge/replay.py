# 실제 공장 CSV(IMDELD 변압기 1초 데이터) 재생
# 전력(kW)과 전압 흔들림 비율을 1초씩 꺼내서 simulator 에 넘겨줌

import numpy as np
import pandas as pd


class Replay:

    # 원본 공장은 수십 kW 규모라, 가동 중 상위 5% 전력이 p95_kw 가 되도록 배율을 곱함 (배율은 전체 데이터 기준)
    # 전압은 고압측(약 6.8kV) 값이라 중앙값 대비 비율만 씀
    # start_date 가 있으면 그 날짜부터만 재생 (학습한 날을 재생하지 않게)
    def __init__(self, path, p95_kw, hour, minute, start_date=None):
        print(f"재생 데이터 로드 중: {path}")
        df = pd.read_csv(path, usecols=["timestamp", "active_power", "voltage"],
                         dtype={"active_power": "float32", "voltage": "float32"})

        # "2017-12-11 18:43:52-02" -> 앞 19글자만 씀 (시간대 표기 버림)
        ts = pd.to_datetime(df["timestamp"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")

        kw = (df["active_power"].clip(lower=0) / 1000).to_numpy()
        running = kw[kw > kw.max() * 0.05]       # 공장 쉬는 시간 빼고
        self.scale = p95_kw / np.percentile(running, 95)
        self.kw = (kw * self.scale).astype("float32")

        v = df["voltage"].to_numpy()
        v_med = np.median(v[v > 0])
        self.v_ratio = np.where(v > 0, v / v_med, 1.0).astype("float32")
        self.ts = ts.to_numpy()

        # 재생 시작 범위 (끝까지 가면 여기로 돌아옴)
        self.first = 0
        if start_date:
            idx = np.flatnonzero(ts >= pd.Timestamp(start_date))
            if len(idx) > 0:
                self.first = int(idx[0])
            else:
                print(f"{start_date} 이후 데이터 없음 -> 전체 재생")

        # 범위 안에서 지금 시:분과 같은 첫 행부터 시작
        minutes = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        idx = np.flatnonzero(minutes == hour * 60 + minute)
        idx = idx[idx >= self.first]
        if len(idx) > 0:
            self.i = int(idx[0])
        else:
            self.i = self.first
        del df

        print(f"재생 준비 완료: {len(self.kw):,}행 · 배율 x{self.scale:.1f} · "
              f"범위 {pd.Timestamp(self.ts[self.first]):%Y-%m-%d}~{pd.Timestamp(self.ts[-1]):%Y-%m-%d} · "
              f"시작 {pd.Timestamp(self.ts[self.i]):%Y-%m-%d %H:%M:%S}")

    # 1초치 꺼내기. 끝까지 가면 시작 범위로 되돌아감
    def next(self):
        kw = float(self.kw[self.i])
        v_ratio = float(self.v_ratio[self.i])
        src_ts = pd.Timestamp(self.ts[self.i])

        self.i += 1
        if self.i >= len(self.kw):
            self.i = self.first
        return kw, v_ratio, src_ts
