import sys

from pymodbus.client import ModbusTcpClient
from config import MODBUS_HOST, MODBUS_PORT, REG


# 레지스터 30(scenario)에 코드를 기록해 시뮬레이터의 이상 상황을 트리거한다.
# Usage:
#     python scenario.py peak   # 피크 부하 (+450kW, 8분)
#     python scenario.py sag    # 전압강하 (-13%, 1분)
#     python scenario.py stop   # 진행 중 시나리오 해제

CODES = {"peak": 1, "sag": 2, "stop": 0}

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CODES:
        print(f"usage: python scenario.py [{'|'.join(CODES)}]")
        sys.exit(1)

    name = sys.argv[1]
    c = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    if not c.connect():
        print(f"접속 실패: {MODBUS_HOST}:{MODBUS_PORT}")
        sys.exit(1)

    # single=True 서버이므로 unit id는 임의 값으로 동일 장비에 매핑됨
    r = c.write_register(REG["scenario"], CODES[name], slave=1)
    c.close()

    if r.isError():
        print("Scenario 전송 실패:", r)
        sys.exit(1)
    print("Scenario 전송:", name)


if __name__ == "__main__":
    main()