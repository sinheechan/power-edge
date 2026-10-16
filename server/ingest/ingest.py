# 수신기: MQTT 수신 -> 품질 체크 -> DB 저장
# 토픽: site/<현장>/meter/data, demand, control, status

import json
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt
import psycopg

MQTT_HOST = "mosquitto"   # docker compose 서비스 이름
V_MIN = 342               # 380V -10%
V_MAX = 418               # 380V +10%
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
NO_DATA_SEC = 30          # 30초 동안 안 오면 수신 끊김
DELAY_SEC = 10            # 측정 후 10초 넘어 도착하면 지연
COOLDOWN = 60             # 같은 알림은 60초에 한 번만

conn = psycopg.connect(os.environ["DB_DSN"], autocommit=True)

last_rx = None       # 마지막으로 받은 시각
last_ts = None       # 마지막 측정 시각
last_kwh = None      # 마지막 누적 kWh
no_data = False      # 지금 수신 끊김 상태인지
last_alert = {}      # 알림 종류별 마지막 기록 시각


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def alert(kind, value, message, ts=None):
    t = now()
    if kind in last_alert and (t - last_alert[kind]).total_seconds() < COOLDOWN:
        return
    last_alert[kind] = t

    conn.execute("INSERT INTO ops.alert VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                 (ts or t, kind, value, message))
    print(f"[알림] {kind}: {message}")


# 계측값 저장 + 품질 체크
def on_data(p):
    global last_rx, last_ts, last_kwh, no_data

    rx = now()
    ts = datetime.fromisoformat(p["ts"])
    buffered = bool(p.get("buffered"))

    last_rx = rx
    if no_data:
        no_data = False
        alert("data_resumed", 0, "데이터 수신 재개")

    cur = conn.execute("""
        INSERT INTO raw.meter_reading
          (ts, seq, voltage, current, kw, pf, hz, kwh, soc, ess_kw, buffered, received_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ts) DO NOTHING""",
        (ts, p["seq"], p["voltage"], p["current"], p["kw"], p["pf"], p["hz"],
         p["kwh"], p["soc"], p["ess_kw"], buffered, rx))

    if cur.rowcount == 0:
        alert("duplicate", p["seq"], "중복 메시지 무시", ts)
        return

    # 버퍼에서 재전송된 건 늦게 오는 게 정상이라 저장만 함
    if buffered:
        return

    delay = (rx - ts).total_seconds()
    if delay > DELAY_SEC:
        alert("delay", delay, f"수신 지연 {delay:.0f}초", ts)

    if last_ts is not None and ts > last_ts and p["kwh"] < last_kwh:
        alert("kwh_backward", p["kwh"], f"누적 전력량 역행 {last_kwh} → {p['kwh']}", ts)
    if last_ts is None or ts > last_ts:
        last_ts = ts
        last_kwh = p["kwh"]

    if not V_MIN <= p["voltage"] <= V_MAX:
        alert("voltage", p["voltage"], f"전압 {p['voltage']:.0f}V 범위 이탈", ts)
    if not HZ_MIN <= p["hz"] <= HZ_MAX:
        alert("frequency", p["hz"], f"주파수 {p['hz']:.2f}Hz 범위 이탈", ts)
    if p["pf"] < PF_MIN:
        alert("power_factor", p["pf"], f"역률 {p['pf']:.2f} 저하", ts)


# 15분 구간 요약 저장. 마감(final)된 구간은 덮어쓰지 않음
def on_demand(p):
    conn.execute("""
        INSERT INTO ops.demand_15min AS d
          (window_start, avg_kw, projected_kw, max_projected_kw, target_kw, exceeded, final, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (window_start) DO UPDATE SET
          avg_kw = EXCLUDED.avg_kw,
          projected_kw = EXCLUDED.projected_kw,
          max_projected_kw = EXCLUDED.max_projected_kw,
          target_kw = EXCLUDED.target_kw,
          exceeded = EXCLUDED.exceeded,
          final = EXCLUDED.final,
          updated_at = EXCLUDED.updated_at
        WHERE NOT d.final""",
        (datetime.fromisoformat(p["window_start"]), p["avg_kw"], p["projected_kw"],
         p["max_projected_kw"], p["target_kw"], p["exceeded"], p["final"], now()))

    if p["final"] and p["exceeded"]:
        alert("peak_exceeded", p["avg_kw"], f"15분 평균 {p['avg_kw']:,.0f}kW 목표 초과")


def on_control(p):
    conn.execute("INSERT INTO ops.control_event VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                 (datetime.fromisoformat(p["ts"]), p["action"], p["target"], p["reason"], p["projected_kw"]))


def on_status(p, site):
    conn.execute("INSERT INTO ops.device_status VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                 (now(), site, p["status"]))
    if p["status"] == "offline":
        alert("edge_offline", 0, f"{site} 엣지 연결 끊김")


# 메시지가 아예 안 오면 on_data 가 안 불려서, 따로 5초마다 확인
def watchdog():
    global no_data

    while True:
        time.sleep(5)
        if last_rx is None or no_data:
            continue

        gap = (now() - last_rx).total_seconds()
        if gap > NO_DATA_SEC:
            no_data = True
            alert("no_data", gap, f"{gap:.0f}초째 데이터 없음")


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT 연결:", reason_code)
    client.subscribe("site/+/#", qos=1)   # 모든 현장의 모든 토픽


# site/plant1/meter/data -> site = "plant1", kind = "meter/data"
def on_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/", 2)
        site = parts[1]
        kind = parts[2]
        p = json.loads(msg.payload)

        if kind == "meter/data":
            on_data(p)
        elif kind == "demand":
            on_demand(p)
        elif kind == "control":
            on_control(p)
        elif kind == "status":
            on_status(p, site)
        else:
            print("모르는 토픽:", msg.topic)
    except psycopg.OperationalError as e:
        # DB 끊기면 종료 -> Docker 가 재시작. 그동안 메시지는 브로커가 보관
        print("DB 연결 오류, 재시작:", e)
        os._exit(1)
    except Exception as e:
        print("처리 실패:", msg.topic, e)


def main():
    threading.Thread(target=watchdog, daemon=True).start()

    # clean_session=False -> 수신기가 꺼져 있는 동안 온 메시지를 브로커가 보관
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ingest", clean_session=False)
    client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(MQTT_HOST, 1883)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
