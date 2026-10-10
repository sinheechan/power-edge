# 엣지 제어기: 계측기 읽기 -> 15분 수요 계산 -> ESS/부하 제어 -> MQTT 전송

import json
import time

import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

from buffer import Buffer
from config import (BUFFER_PATH, ESS_MAX_KW, LOADS, MODBUS_HOST, MODBUS_PORT, MQTT_HOST, MQTT_PASS,
                    MQTT_PORT, MQTT_USER, REG, SHED_ORDER, SITE, TARGET_KW, now_kst)
from demand import Demand

BASE = f"site/{SITE}"

WARN_RATIO = 0.9
SAFE_RATIO = 0.8
SOC_MIN = 20
HOLD_MIN = 2
COOLDOWN_SEC = 60


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
        "kwh": ((h[5] << 16) | h[6]) / 10,   # 32bit (상위/하위)
        "soc": h[10] / 10,
        "ess_kw": h[11] / 10,
    }


# 15분 예상 수요 기준으로 ESS 방전 / 부하 차단 / 복귀
class Control:

    def __init__(self, mb):
        self.mb = mb
        self.shed = []
        self.ess_on = False
        self.last = None

    # 이전 실행에서 차단된 채 남아 있을 수 있어서 시작 시 초기화
    def reset(self):
        for addr in LOADS.values():
            self.mb.write_coil(addr, True, slave=1)
        self.mb.write_register(REG["ess_cmd"], 0, slave=1)

    def make_event(self, t, action, target, reason, p):
        self.last = t
        return {
            "ts": t,
            "action": action,
            "target": target,
            "reason": reason,
            "projected_kw": round(p, 1),
        }

    # 한 번에 하나만 동작. 동작 시 이벤트 dict 반환
    def step(self, t, d, soc):
        if d.elapsed_min < HOLD_MIN:
            return None
        if self.last is not None and (t - self.last).total_seconds() < COOLDOWN_SEC:
            return None

        p = d.projected
        target = d.target

        if p >= target and len(self.shed) < len(SHED_ORDER) and (self.ess_on or soc <= SOC_MIN):
            name = SHED_ORDER[len(self.shed)]
            self.mb.write_coil(LOADS[name], False, slave=1)
            self.shed.append(name)
            return self.make_event(t, "shed", name, "예상 수요 ≥ 목표", p)

        if p >= target * WARN_RATIO and not self.ess_on and soc > SOC_MIN:
            self.mb.write_register(REG["ess_cmd"], int(ESS_MAX_KW * 10), slave=1)
            self.ess_on = True
            return self.make_event(t, "ess_on", "ess", "예상 수요 ≥ 목표 90%", p)

        if p < target * SAFE_RATIO and self.shed:
            name = self.shed.pop()
            self.mb.write_coil(LOADS[name], True, slave=1)
            return self.make_event(t, "restore", name, "예상 수요 < 목표 80%", p)

        if self.ess_on and (p < target * SAFE_RATIO or soc <= SOC_MIN):
            self.mb.write_register(REG["ess_cmd"], 0, slave=1)
            self.ess_on = False
            reason = "여유 확보" if soc > SOC_MIN else "충전율 부족"
            return self.make_event(t, "ess_off", "ess", reason, p)

        return None


# MQTT 전송. 끊기면 버퍼에 저장했다가 재연결 시 재전송
class Link:

    def __init__(self):
        self.buf = Buffer(BUFFER_PATH)

        self.cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-{SITE}")
        self.cli.username_pw_set(MQTT_USER, MQTT_PASS)
        self.cli.will_set(f"{BASE}/status", json.dumps({"status": "offline"}), qos=1, retain=True)   # LWT
        self.cli.on_connect = self.on_connect
        self.cli.reconnect_delay_set(1, 10)
        self.cli.connect_async(MQTT_HOST, MQTT_PORT, keepalive=10)
        self.cli.loop_start()

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print("MQTT 연결:", reason_code)
        if not reason_code.is_failure:
            client.publish(f"{BASE}/status", json.dumps({"status": "online"}), qos=1, retain=True)

    def send(self, sub, payload):
        topic = f"{BASE}/{sub}"

        if self.cli.is_connected():
            info = self.cli.publish(topic, json.dumps(payload, default=str), qos=1)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                return

        payload["buffered"] = True
        self.buf.put(topic, json.dumps(payload, default=str))

    def flush(self):
        if not self.cli.is_connected():
            return

        done = []
        for i, topic, payload in self.buf.take():
            result = self.cli.publish(topic, payload, qos=1)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                break
            done.append(i)

        if done:
            self.buf.delete(done)


def main():
    mb = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    link = Link()
    dem = Demand(TARGET_KW)
    ctl = Control(mb)

    seq = 0
    ready = False

    while True:
        t0 = time.time()
        t = now_kst()

        try:
            if not mb.connected:
                mb.connect()
            if not ready:
                ctl.reset()
                ready = True
            m = read_meter(mb)
        except Exception as e:
            print("Modbus 읽기 실패:", e)
            time.sleep(2)
            continue

        seq += 1
        data = {"ts": t, "seq": seq}
        data.update(m)
        link.send("meter/data", data)

        closed = dem.update(t, m["kwh"], m["kw"])
        if closed:
            link.send("demand", closed)
        if seq % 5 == 0:
            link.send("demand", dem.summary())

        try:
            ev = ctl.step(t, dem, m["soc"])
        except Exception as e:
            print("제어 명령 실패:", e)
            ev = None
        if ev:
            print("[제어]", ev)
            link.send("control", ev)

        link.flush()

        if seq % 10 == 0:
            mqtt_state = "연결" if link.cli.is_connected() else "끊김"
            print(f"{t:%H:%M:%S} | {m['kw']:,.0f} kW | 15분 예상 {dem.projected:,.0f}/{TARGET_KW:,.0f} kW "
                  f"| ESS {m['soc']:.0f}% | 버퍼 {link.buf.size()} | MQTT {mqtt_state}")

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1 - elapsed))


if __name__ == "__main__":
    main()