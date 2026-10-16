# 15분 수요 계산 (한전 최대수요전력 = 15분 평균 전력)
# 구간: 매시 00, 15, 30, 45분 시작
# 예상 평균(kW) = (지금까지 쓴 kWh + 지금 kW x 남은 시간) / 0.25시간


class Demand:

    def __init__(self, target):
        self.target = target     # 목표 kW
        self.start = None        # 지금 구간 시작 시각
        self.kwh0 = 0.0          # 구간 시작 때 누적 kWh
        self.used = 0.0          # 구간 안에서 쓴 kWh
        self.elapsed_min = 0.0   # 구간 시작 후 지난 분
        self.projected = 0.0     # 구간 끝날 때 예상 평균 kW
        self.max_proj = 0.0      # 구간 안에서 가장 높았던 예상치

    # 매초 호출. 구간이 바뀌었으면 끝난 구간 요약을 돌려줌
    def update(self, t, kwh, kw):
        closed = None
        ws = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)

        if self.start != ws:
            if self.start is None:
                # 처음 켰을 때: 구간 시작부터 지금 kW 로 써왔다고 가정
                passed_h = (t - ws).total_seconds() / 3600
                self.kwh0 = kwh - kw * passed_h
            else:
                # 구간 바뀜: 끝난 구간 마감
                self.used = kwh - self.kwh0
                closed = self.summary(final=True)
                self.kwh0 = kwh
            self.start = ws
            self.max_proj = 0.0

        elapsed_h = (t - self.start).total_seconds() / 3600
        remain_h = 0.25 - elapsed_h
        self.elapsed_min = elapsed_h * 60
        self.used = kwh - self.kwh0
        self.projected = (self.used + kw * remain_h) / 0.25

        if self.projected > self.max_proj:
            self.max_proj = self.projected

        return closed

    # 구간 요약 (MQTT 로 보냄). final = 구간 마감
    def summary(self, final=False):
        if final:
            avg = self.used / 0.25
            proj = avg
        else:
            if self.elapsed_min > 0:
                avg = self.used / (self.elapsed_min / 60)
            else:
                avg = 0.0
            proj = self.projected

        return {
            "window_start": self.start,
            "avg_kw": round(avg, 1),
            "projected_kw": round(proj, 1),
            "max_projected_kw": round(self.max_proj, 1),
            "target_kw": self.target,
            "exceeded": proj >= self.target,
            "final": final,
        }
