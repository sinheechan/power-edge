"""
Virtual power meter / ESS simulator (Modbus TCP server).

1초 주기로 공장 부하·ESS·계측값을 산출해 홀딩 레지스터에 기록한다.
엣지 제어기(controller.py)는 Modbus 클라이언트로 접속해 값을 폴링하고,
코일(부하 차단)·레지스터 20(ESS 방전 지령)을 통해 제어 명령을 내린다.

Register map: config.REG 참조
Scenario   : 레지스터 30에 코드 기록 시 동작 (1=피크, 2=전압강하, 0=해제)
"""
import asyncio
import math
import os
import random
from datetime import timedelta

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
from pymodbus.server import StartAsyncTcpServer

from config import LOAD_KW, LOADS, MODBUS_PORT, REG, now_kst

ESS_CAP_KWH = 500                  # ESS 정격 용량 (kWh)
SCENARIO_SEC = {1: 480, 2: 60}     # 시나리오 코드별 지속 시간 (s)
PEAK_EXTRA_KW = 450                # 피크 시나리오 추가 부하 (kW)
SAG_RATIO = 0.87                   # 전압강하 시나리오 전압 비율
NOMINAL_V = 380                    # 3상 공칭 전압 (V)
SIM_HOUR = os.getenv("SIM_HOUR")   # 부하 프로파일 시각 고정 (테스트용, 미설정 시 실제 시각)
DEBUG_PRINT = True                 # 주기별 산출값 콘솔 출력

# 시간대별 비차단 기본 부하 (kW). SIM_HOUR 설정 시 해당 시각으로 고정.
def base_kw(t):
    h = int(SIM_HOUR) if SIM_HOUR else t.hour

    if h == 12:                        # 점심
        return 800
    if 8 <= h < 18:                    # 주간 가동
        return 1000
    if 6 <= h < 8 or 18 <= h < 20:     # 가동 전후
        return 700
    return 450                         # 야간


# 1초 주기 상태 갱신 루프.
# 1. 시나리오 상태 갱신
# 2. 부하 산출 (기본 부하 + 운전 중 차단 가능 부하 + 시나리오 가산)
# 3. ESS 방전 반영 및 SOC 갱신 → 계통 수전 전력(grid) 산출
# 4. 전압·전류·역률·주파수·누적 전력량 산출
# 5. 레지스터 기록
async def update(ctx):
    s = ctx[0]
    s.setValues(1, 0, [1, 1, 1])       # 초기 상태: 전 부하 운전
    kwh, soc, scen, scen_end = 0.0, 80.0, 0, None

    while True:
        t = now_kst()

        # 시나리오: 코드 변경 감지 시 시작/해제, 지속 시간 경과 시 자동 해제
        code = s.getValues(3, REG["scenario"], 1)[0]
        if code != scen:
            scen = code
            scen_end = t + timedelta(seconds=SCENARIO_SEC.get(code, 0))
            print(f"[{t:%H:%M:%S}] 시나리오 {code if code else '종료'}")
        if scen and t >= scen_end:
            scen = 0
            s.setValues(3, REG["scenario"], [0])
            print(f"[{t:%H:%M:%S}] 시나리오 종료")

        # 부하: 코일 상태(제어기 차단 여부) 반영, ±1% 계측 노이즈
        coils = s.getValues(1, 0, 3)
        load = base_kw(t) + sum(LOAD_KW[n] for n, a in LOADS.items() if coils[a])
        if scen == 1:
            load += PEAK_EXTRA_KW
        load *= 1 + random.gauss(0, 0.01)

        # ESS: 방전만 모델링, 대기 시 저속 충전으로 단순화. SOC 5% 이하 방전 불가
        cmd = s.getValues(3, REG["ess_cmd"], 1)[0] / 10
        ess = cmd if soc > 5 else 0.0
        if ess > 0:
            soc -= ess / ESS_CAP_KWH * 100 / 3600
        else:
            soc = min(100.0, soc + 0.005)
        grid = max(load - ess, 0.0)    # 계통 수전 전력 = 부하 − ESS 방전

        # 계측값: 3상 전류 I = P / (√3 · V · PF), 누적 전력량은 1초분 적산
        v = NOMINAL_V * (1 + random.gauss(0, 0.003)) * (SAG_RATIO if scen == 2 else 1)
        pf = min(0.99, 0.95 + random.gauss(0, 0.005))
        hz = 60 + random.gauss(0, 0.01)
        amp = grid * 1000 / (math.sqrt(3) * v * pf)
        kwh += grid / 3600
        k = int(kwh * 10)

        # 레지스터 기록 (배율 적용). kWh는 32bit → 상위/하위 16bit 분할
        s.setValues(3, 0, [int(v * 10), int(amp * 10), int(grid * 10), int(pf * 1000),
                           int(hz * 100), k >> 16, k & 0xFFFF])
        s.setValues(3, REG["soc"], [int(soc * 10), int(ess * 10)])

        if DEBUG_PRINT:
            print(f"[{t:%H:%M:%S}] 부하 {load:7.1f}kW | ESS {ess:5.1f}kW | 한전 {grid:7.1f}kW | "
                  f"{v:5.1f}V {amp:7.1f}A 역률 {pf:.3f} {hz:.2f}Hz | 누적 {kwh:8.2f}kWh | "
                  f"SOC {soc:5.1f}% | 스위치 {[int(x) for x in coils]} | 시나리오 {scen}")

        await asyncio.sleep(1) # 대기 중 & Modbus 서버가 클라이언트 요청 처리


async def main():
    """데이터스토어 초기화 → 갱신 루프 기동 → Modbus TCP 서버 실행."""
    store = ModbusSlaveContext(
        co=ModbusSequentialDataBlock(0, [0] * 10),   # 코일 10
        hr=ModbusSequentialDataBlock(0, [0] * 50),   # 홀딩 레지스터 50
    )
    ctx = ModbusServerContext(slaves=store, single=True)
    task = asyncio.create_task(update(ctx))          # 참조 유지 (GC 방지)
    print(f"가상 계측기 실행: 0.0.0.0:{MODBUS_PORT}")
    await StartAsyncTcpServer(context=ctx, address=("0.0.0.0", MODBUS_PORT))


if __name__ == "__main__":
    asyncio.run(main())