-- 스키마 = 테이블 묶음. raw: 받은 그대로, ops: 운영 기록
CREATE SCHEMA raw;
CREATE SCHEMA ops;

-- 1초 계측값. ts(측정 시각)가 기본키 → 같은 시각 데이터가 두 번 와도 한 번만 저장
CREATE TABLE raw.meter_reading (
  ts TIMESTAMP PRIMARY KEY,
  seq INT,                          -- 엣지가 붙인 순번
  voltage REAL, current REAL, kw REAL, pf REAL, hz REAL,
  kwh DOUBLE PRECISION,             -- 누적값이라 커지므로 정밀도 높은 타입
  soc REAL, ess_kw REAL,
  buffered BOOLEAN DEFAULT false,   -- 버퍼에서 늦게 재전송된 데이터인지
  received_at TIMESTAMP             -- 서버가 받은 시각 (지연 계산용)
);

-- 15분 구간 요약. 진행 중엔 계속 덮어쓰고, final=true(마감)가 되면 고정
CREATE TABLE ops.demand_15min (
  window_start TIMESTAMP PRIMARY KEY,
  avg_kw REAL, projected_kw REAL, max_projected_kw REAL, target_kw REAL,
  exceeded BOOLEAN, final BOOLEAN, updated_at TIMESTAMP
);

-- 제어 이력 (ESS 방전·정지, 부하 차단·복귀)
CREATE TABLE ops.control_event (
  ts TIMESTAMP, action TEXT, target TEXT, reason TEXT, projected_kw REAL,
  PRIMARY KEY (ts, action, target)
);

-- 알림 (끊김, 지연, 범위 이탈, 피크 초과 등)
CREATE TABLE ops.alert (ts TIMESTAMP, kind TEXT, value REAL, message TEXT, PRIMARY KEY (ts, kind));

-- 현장 연결 상태 변화 (online / offline)
CREATE TABLE ops.device_status (ts TIMESTAMP, site TEXT, status TEXT, PRIMARY KEY (ts, site));