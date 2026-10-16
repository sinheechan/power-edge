# 엣지 제어기: 계측기 읽기 -> 15분 수요 계산 -> ESS/부하 제어 -> MQTT 전송
# MQTT 가 끊기면 buffer.db 에 쌓아뒀다가 다시 연결되면 보냄

import json
import sqlite3
import time

import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

from config import (ESS_KW, MODBUS_HOST, MODBUS_PORT, MQTT_HOST, MQTT_PASS, MQTT_PORT, MQTT_USER,
                    SHED_ORDER, SITE, TARGET_KW, now_kst)
from demand import Demand

TOPIC = f"site/{SITE}"
WARN = 0.9      # 예상 수요가 목표의 90% 넘으면 ESS 방전
SAFE = 0.8      # 목표의 80% 밑으로 내려가면 복귀
SOC_MIN = 20    # ESS 충전율 20% 이하면 방전 안 함
HOLD_MIN = 2    # 구간 시작 2분은 예상치가 불안정해서 판단 안 함
COOLDOWN = 60   # 한 번 제어하면 60초 동안 다음 제어 안 함

# 못 보낸 메시지 저장용 (파일이라 프로그램 재시작해도 남음)
db = sqlite3.connect("buffer.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS buf (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, payload TEXT)")
db.commit()

# 제어 상태
shed = []            # 차단한 부하 번호 (차단한 순서대로)
ess_on = False
last_action = None   # 마지막 제어 시각


def read_meter(mb):
    r = mb.read_holding_registers(0, 12, slave=1)
    if r.isError():
        raise IOError(r)

    h = r.registers
    return {
        "voltage": h[0] / 10,
        "current": h[1] / 10,
        "kw": h[2] / 10,
        "pf": h[3] / 1000,
        "hz": h[4] / 100,
        "kwh": ((h[5] << 16) | h[6]) / 10,
        "soc": h[10] / 10,
        "ess_kw": h[11] / 10,
    }


# 15분 예상 수요 보고 한 번에 하나만 제어. 제어했으면 기록용 dict 반환
#   목표 이상 + ESS 이미 방전 중(또는 충전율 부족) -> 부하 차단
#   목표 90% 이상                                   -> ESS 방전
#   목표 80% 미만                                   -> 부하 복귀, 그다음 ESS 정지
def control(mb, t, dem, soc):
    global ess_on, last_action

    if dem.elapsed_min < HOLD_MIN:
        return None
    if last_action is not None and (t - last_action).total_seconds() < COOLDOWN:
        return None

    p = dem.projected
    if p >= TARGET_KW and len(shed) < 3 and (ess_on or soc <= SOC_MIN):
        n = SHED_ORDER[len(shed)]
        mb.write_coil(n, False, slave=1)
        shed.append(n)
        action, target, reason = "shed", f"load{n + 1}", "예상 수요 ≥ 목표"
    elif p >= TARGET_KW * WARN and not ess_on and soc > SOC_MIN:
        mb.write_register(20, ESS_KW * 10, slave=1)
        ess_on = True
        action, target, reason = "ess_on", "ess", "예상 수요 ≥ 목표 90%"
    elif p < TARGET_KW * SAFE and shed:
        n = shed.pop()
        mb.write_coil(n, True, slave=1)
        action, target, reason = "restore", f"load{n + 1}", "예상 수요 < 목표 80%"
    elif ess_on and (p < TARGET_KW * SAFE or soc <= SOC_MIN):
        mb.write_register(20, 0, slave=1)
        ess_on = False
        action, target = "ess_off", "ess"
        reason = "여유 확보" if soc > SOC_MIN else "충전율 부족"
    else:
        return None

    last_action = t
    return {"ts": t, "action": action, "target": target, "reason": reason, "projected_kw": round(p, 1)}


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT 연결:", reason_code)
    if not reason_code.is_failure:
        client.publish(f"{TOPIC}/status", json.dumps({"status": "online"}), qos=1, retain=True)


# MQTT 전송. 못 보내면 buffer.db 에 저장 (default=str: 시각을 문자열로 바꿔서 보냄)
def send(cli, sub, payload):
    topic = f"{TOPIC}/{sub}"
    if cli.is_connected():
        r = cli.publish(topic, json.dumps(payload, default=str), qos=1)
        if r.rc == mqtt.MQTT_ERR_SUCCESS:
            return

    payload["buffered"] = True
    db.execute("INSERT INTO buf (topic, payload) VALUES (?, ?)", (topic, json.dumps(payload, default=str)))
    db.commit()


# 쌓인 메시지를 오래된 것부터 200개씩 재전송. 보낸 것만 지움
def flush(cli):
    if not cli.is_connected():
        return

    rows = db.execute("SELECT id, topic, payload FROM buf ORDER BY id LIMIT 200").fetchall()
    for i, topic, payload in rows:
        r = cli.publish(topic, payload, qos=1)
        if r.rc != mqtt.MQTT_ERR_SUCCESS:
            break
        db.execute("DELETE FROM buf WHERE id = ?", (i,))
    db.commit()


def main():
    mb = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-{SITE}")
    cli.username_pw_set(MQTT_USER, MQTT_PASS)
    # 엣지가 갑자기 끊기면 브로커가 대신 offline 을 보내줌
    cli.will_set(f"{TOPIC}/status", json.dumps({"status": "offline"}), qos=1, retain=True)
    cli.on_connect = on_connect
    cli.reconnect_delay_set(1, 10)
    cli.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)
    cli.loop_start()

    dem = Demand(TARGET_KW)
    seq = 0
    ready = False

    while True:
        t0 = time.time()
        t = now_kst()

        try:
            if not mb.connected:
                mb.connect()
            if not ready:
                # 지난번 실행에서 차단된 채 남아 있을 수 있어서 처음에 전부 복귀
                for n in range(3):
                    mb.write_coil(n, True, slave=1)
                mb.write_register(20, 0, slave=1)
                ready = True
            meter = read_meter(mb)
        except Exception as e:
            print("Modbus 읽기 실패:", e)
            time.sleep(2)
            continue

        seq += 1
        data = {"ts": t, "seq": seq}
        data.update(meter)
        send(cli, "meter/data", data)

        closed = dem.update(t, meter["kwh"], meter["kw"])
        if closed:
            send(cli, "demand", closed)
        if seq % 5 == 0:
            send(cli, "demand", dem.summary())

        try:
            event = control(mb, t, dem, meter["soc"])
        except Exception as e:
            print("제어 명령 실패:", e)
            event = None
        if event:
            print("[제어]", event)
            send(cli, "control", event)

        flush(cli)

        if seq % 10 == 0:
            n_buf = db.execute("SELECT count(*) FROM buf").fetchone()[0]
            state = "연결" if cli.is_connected() else "끊김"
            print(f"{t:%H:%M:%S} | {meter['kw']:,.0f} kW | 15분 예상 {dem.projected:,.0f}/{TARGET_KW:,.0f} kW "
                  f"| ESS {meter['soc']:.0f}% | 버퍼 {n_buf} | MQTT {state}")

        time.sleep(max(0.0, 1 - (time.time() - t0)))


if __name__ == "__main__":
    main()
