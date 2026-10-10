# 한국 전기요금의 최대수요전력 기준(15분 평균 전력)에 맞춰
# 현재 구간의 사용량과 구간 종료 시점 예상 평균(projected)을 산출한다.
#     projected (kW) = (구간 내 사용 kWh + 현재 kW × 잔여 시간 h) / 0.25 h
# 구간 경계: 매시 00 · 15 · 30 · 45분

WINDOW_H = 0.25


# 15분 구간 수요 상태를 유지하며 매 주기 갱신
class Demand:

    def __init__(self, target):
        self.target = target       # 목표 최대수요전력 (kW)
        self.start = None          # 현재 구간 시작 시각
        self.kwh0 = 0.0            # 구간 시작 시점 누적 전력량 (kWh)
        self.used = 0.0            # 구간 내 사용 전력량 (kWh)
        self.elapsed_min = 0.0     # 구간 경과 시간 (min)
        self.projected = 0.0       # 구간 종료 시점 예상 평균 (kW)
        self.max_proj = 0.0        # 구간 내 예상치 최댓값 (kW)

    def update(self, t, kwh, kw):
        closed = None
        ws = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)

        # 구간 전환 처리
        if self.start != ws:
            if self.start is None:
                # 최초 기동
                passed_h = (t - ws).total_seconds() / 3600
                self.kwh0 = kwh - kw * passed_h
            else:
                # 신규 구간 시작값 기록
                self.used = kwh - self.kwh0
                closed = self.summary(final=True)
                self.kwh0 = kwh
            self.start = ws
            self.max_proj = 0.0

        # 현재 구간 산출
        elapsed_h = (t - self.start).total_seconds() / 3600
        remain_h = WINDOW_H - elapsed_h
        self.elapsed_min = elapsed_h * 60
        self.used = kwh - self.kwh0
        self.projected = (self.used + kw * remain_h) / WINDOW_H

        if self.projected > self.max_proj:
            self.max_proj = self.projected

        return closed

    # 구간 요약 생성
    def summary(self, final=False):
        if final:
            avg = self.used / WINDOW_H
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