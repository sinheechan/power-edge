# 가상 계측기 + ESS (Modbus TCP 서버)
# 1초마다 부하, ESS, 전압/전류를 계산해서 레지스터에 써둠
# controller.py 가 접속해서 값을 읽고, 코일(부하 차단)과 20번 레지스터(ESS 지령)로 제어함
# 시나리오: 30번 레지스터에 1 = 피크, 2 = 전압강하, 0 = 해제

import asyncio
import math
import random
from datetime import timedelta

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
from pymodbus.server import StartAsyncTcpServer

from config import LOAD_KW, MODBUS_PORT, now_kst

# 부하 데이터: 실제 공장 CSV 재생. CSV_PATH = None 이면 시간대별 합성 부하
CSV_PATH = "../data/mvlv-transformer.csv"
P95_KW = 1400               # 재생 데이터 규모 (가동 중 상위 5% 전력)
REPLAY_FROM = "2018-03-29"  # 이 날짜부터 재생 (학습에 안 쓴 test 기간). None 이면 전체
SIM_HOUR = 10               # 시각 고정 (테스트용). None 이면 지금 시각

ESS_KWH = 500       # ESS 용량
PEAK_KW = 450       # 피크 시나리오: 부하 +450kW
PEAK_SEC = 480      #                8분 동안
SAG_RATIO = 0.87    # 전압강하 시나리오: 전압 -13%
SAG_SEC = 60        #                    1분 동안
VOLT = 380          # 공칭 전압
PRINT = True        # 1초마다 값 출력


# 합성 부하 모드일 때 시간대별 기본 부하 (차단 못 하는 부하)
def base_kw(t):
    h = t.hour
    if SIM_HOUR is not None:
        h = SIM_HOUR

    if h == 12:                       # 점심
        return 800
    if 8 <= h < 18:                   # 주간
        return 1000
    if 6 <= h < 8 or 18 <= h < 20:    # 출근 전, 퇴근 후
        return 700
    return 450                        # 야간


# 1초마다 상태 계산해서 레지스터에 기록
async def update(ctx, replay):
    dev = ctx[0]
    dev.setValues(1, 0, [1, 1, 1])    # 코일: 부하 3개 모두 운전
    kwh = 0.0
    soc = 80.0
    scenario = 0
    scenario_end = None

    while True:
        t = now_kst()

        # 시나리오: 번호가 바뀌면 시작, 시간 다 되면 자동 해제
        code = dev.getValues(3, 30, 1)[0]
        if code != scenario:
            scenario = code
            if code == 1:
                sec = PEAK_SEC
            elif code == 2:
                sec = SAG_SEC
            else:
                sec = 0
            scenario_end = t + timedelta(seconds=sec)
            print(f"[{t:%H:%M:%S}] 시나리오 {code if code else '종료'}")
        if scenario and t >= scenario_end:
            scenario = 0
            dev.setValues(3, 30, [0])
            print(f"[{t:%H:%M:%S}] 시나리오 종료")

        # 부하. 코일 0 = controller 가 차단한 부하
        coils = dev.getValues(1, 0, 3)
        v_ratio = 1.0
        src = ""
        if replay is not None:
            # 실제 데이터는 공장 전체 전력이라, 차단된 부하만큼 빼줌
            rep_kw, v_ratio, src_ts = replay.next()
            shed_kw = 0
            for i in range(3):
                if not coils[i]:
                    shed_kw += LOAD_KW[i]
            load = max(rep_kw - shed_kw, 0.0)
            src = f" | 원본 {src_ts:%m-%d %H:%M:%S}"
        else:
            load = base_kw(t)
            for i in range(3):
                if coils[i]:
                    load += LOAD_KW[i]
            load *= 1 + random.gauss(0, 0.01)    # 계측 노이즈 1%
        if scenario == 1:
            load += PEAK_KW

        # ESS: 지령만큼 방전, 쉬는 동안 천천히 충전. 충전율 5% 이하면 방전 안 함
        cmd = dev.getValues(3, 20, 1)[0] / 10
        ess = cmd if soc > 5 else 0.0
        if ess > 0:
            soc -= ess / ESS_KWH * 100 / 3600
        else:
            soc = min(100.0, soc + 0.005)
        grid = max(load - ess, 0.0)      # 한전에서 받는 전력 = 부하 - ESS

        # 전압: 재생이면 실제 흔들림 비율, 합성이면 노이즈 0.3%
        if replay is not None:
            v = VOLT * v_ratio
        else:
            v = VOLT * (1 + random.gauss(0, 0.003))
        if scenario == 2:
            v *= SAG_RATIO
        pf = min(0.99, 0.95 + random.gauss(0, 0.005))
        hz = 60 + random.gauss(0, 0.01)
        amp = grid * 1000 / (math.sqrt(3) * v * pf)    # 3상 전류 = P / (√3 x V x 역률)
        kwh += grid / 3600
        k = int(kwh * 10)

        # 레지스터 0~6, 10~11 기록 (배율은 config.py 주소표 참고)
        dev.setValues(3, 0, [int(v * 10), int(amp * 10), int(grid * 10), int(pf * 1000),
                             int(hz * 100), k >> 16, k & 0xFFFF])
        dev.setValues(3, 10, [int(soc * 10), int(ess * 10)])

        if PRINT:
            print(f"[{t:%H:%M:%S}] 부하 {load:7.1f}kW | ESS {ess:5.1f}kW | 한전 {grid:7.1f}kW | "
                  f"{v:5.1f}V {amp:7.1f}A 역률 {pf:.3f} {hz:.2f}Hz | 누적 {kwh:8.2f}kWh | "
                  f"SOC {soc:5.1f}% | 스위치 {[int(x) for x in coils]} | 시나리오 {scenario}{src}")

        await asyncio.sleep(1)   # 쉬는 동안 Modbus 서버가 요청 처리


async def main():
    replay = None
    if CSV_PATH:
        from replay import Replay   # 재생할 때만 pandas 불러옴
        t = now_kst()
        hour = t.hour
        if SIM_HOUR is not None:
            hour = SIM_HOUR
        replay = Replay(CSV_PATH, P95_KW, hour, t.minute, REPLAY_FROM)
    else:
        print("부하: 시간대별 합성 데이터")

    store = ModbusSlaveContext(
        co=ModbusSequentialDataBlock(0, [0] * 10),   # 코일 10개
        hr=ModbusSequentialDataBlock(0, [0] * 50),   # 레지스터 50개
    )
    ctx = ModbusServerContext(slaves=store, single=True)
    task = asyncio.create_task(update(ctx, replay))   # 변수에 담아둬야 중간에 안 사라짐
    print(f"가상 계측기 실행: 0.0.0.0:{MODBUS_PORT}")
    await StartAsyncTcpServer(context=ctx, address=("0.0.0.0", MODBUS_PORT))


if __name__ == "__main__":
    asyncio.run(main())
