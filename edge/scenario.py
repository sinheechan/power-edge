# 시뮬레이터에 이상 상황 일으키기 (30번 레지스터에 번호를 씀)
#   python scenario.py peak   # 피크 부하 (+450kW, 8분)
#   python scenario.py sag    # 전압강하 (-13%, 1분)
#   python scenario.py stop   # 해제

import sys

from pymodbus.client import ModbusTcpClient

from config import MODBUS_HOST, MODBUS_PORT


def main():
    if len(sys.argv) != 2:
        print("usage: python scenario.py [peak|sag|stop]")
        sys.exit(1)

    name = sys.argv[1]
    if name == "peak":
        code = 1
    elif name == "sag":
        code = 2
    elif name == "stop":
        code = 0
    else:
        print("usage: python scenario.py [peak|sag|stop]")
        sys.exit(1)

    c = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    if not c.connect():
        print(f"접속 실패: {MODBUS_HOST}:{MODBUS_PORT}")
        sys.exit(1)

    r = c.write_register(30, code, slave=1)
    c.close()

    if r.isError():
        print("Scenario 전송 실패:", r)
        sys.exit(1)
    print("Scenario 전송:", name)


if __name__ == "__main__":
    main()
