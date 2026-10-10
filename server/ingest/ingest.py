# 수신기: MQTT 수신 -> 품질 체크 -> DB 저장

import json
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt
import psycopg

V_MIN = 342          # 380V -10%
V_MAX = 418          # 380V +10%
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
NO_DATA_SEC = 30
DELAY_SEC = 10
COOLDOWN_SEC = 60    # 같은 알림 반복 방지

conn = psycopg.connect(os.environ["DB_DSN"], autocommit=True)

state = {
    "last_rx": None,     # 마지막 수신 시각
    "last_ts": None,     # 마지막 측정 시각
    "last_kwh": None,
    "no_data": False,
}
last_alert = {}


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


# 같은 종류 알림은 60초에 한 번만 기록
def alert(kind, value, message, ts=None):
    t = now()
    if kind in last_alert and (t - last_alert[kind]).total_seconds() < COOLDOWN_SEC:
        return
    last_alert[kind] = t

    conn.execute("INSERT INTO ops.alert VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                 (ts or t, kind, value, message))
    print(f"[알림] {kind}: {message}")


# 계측값 저장 + 품질 체크
def on_data(p, site):
    rx = now()
    ts = datetime.fromisoformat(p["ts"])
    buffered = bool(p.get("buffered"))

    state["last_rx"] = rx
    if state["no_data"]:
        state["no_data"] = False
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

    # 재전송분은 늦게 오는 게 정상이라 저장만 함
    if buffered:
        return

    delay = (rx - ts).total_seconds()
    if delay > DELAY_SEC:
        alert("delay", delay, f"수신 지연 {delay:.0f}초", ts)

    last_ts = state["last_ts"]
    if last_ts is not None and ts > last_ts and p["kwh"] < state["last_kwh"]:
        alert("kwh_backward", p["kwh"], f"누적 전력량 역행 {state['last_kwh']} → {p['kwh']}", ts)
    if last_ts is None or ts > last_ts:
        state["last_ts"] = ts
        state["last_kwh"] = p["kwh"]

    if not V_MIN <= p["voltage"] <= V_MAX:
        alert("voltage", p["voltage"], f"전압 {p['voltage']:.0f}V 범위 이탈", ts)
    if not HZ_MIN <= p["hz"] <= HZ_MAX:
        alert("frequency", p["hz"], f"주파수 {p['hz']:.2f}Hz 범위 이탈", ts)
    if p["pf"] < PF_MIN:
        alert("power_factor", p["pf"], f"역률 {p['pf']:.2f} 저하", ts)


# 15분 구간 요약 저장. 마감(final)된 구간은 덮어쓰지 않음
def on_demand(p, site):
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


def on_control(p, site):
    conn.execute("INSERT INTO ops.control_event VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                 (datetime.fromisoformat(p["ts"]), p["action"], p["target"], p["reason"], p["projected_kw"]))


def on_status(p, site):
    conn.execute("INSERT INTO ops.device_status VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                 (now(), site, p["status"]))
    if p["status"] == "offline":
        alert("edge_offline", 0, f"{site} 엣지 연결 끊김")


HANDLERS = {
    "meter/data": on_data,
    "demand": on_demand,
    "control": on_control,
    "status": on_status,
}


# 메시지가 아예 안 오면 on_data 가 안 불리니까 별도 스레드로 감시
def watchdog():
    while True:
        time.sleep(5)
        last = state["last_rx"]
        if last is None or state["no_data"]:
            continue

        gap = (now() - last).total_seconds()
        if gap > NO_DATA_SEC:
            state["no_data"] = True
            alert("no_data", gap, f"{gap:.0f}초째 데이터 없음")


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT 연결:", reason_code)
    client.subscribe("site/+/#", qos=1)   # 모든 현장의 모든 토픽


# site/plant1/meter/data -> site="plant1", kind="meter/data"
def on_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/", 2)
        site = parts[1]
        kind = parts[2]
        HANDLERS[kind](json.loads(msg.payload), site)
    except psycopg.OperationalError as e:
        # DB 끊기면 종료 -> Docker 가 재시작, 그동안 메시지는 브로커가 보관
        print("DB 연결 오류, 재시작:", e)
        os._exit(1)
    except Exception as e:
        print("처리 실패:", msg.topic, e)


def main():
    threading.Thread(target=watchdog, daemon=True).start()

    # clean_session=False -> 수신기 꺼져 있는 동안 메시지를 브로커가 보관
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ingest", clean_session=False)
    client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(os.getenv("MQTT_HOST", "mosquitto"), 1883)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()