import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


# config.py : 설정관리 (모든 pc에서 .env 파일에 대해 모든 설정값을 동일하게 설정하기 위함)


load_dotenv() # 같은 폴더의 .env를 읽어 환경 변수로 등록

# .env 값 읽기. 두 번째 인자는 .env 에 없을 때 쓰는 기본값
MODBUS_HOST = os.getenv("MODBUS_HOST", "localhost")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", 5020))   # 문자열로 읽히므로 숫자로 변환
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
SITE = os.getenv("SITE", "plant1")
TARGET_KW = float(os.getenv("TARGET_KW", 1600))
ESS_MAX_KW = float(os.getenv("ESS_MAX_KW", 200))
BUFFER_PATH = os.getenv("BUFFER_PATH", "buffer.db")  # 오프라인 버퍼 파일 이름

# 홀딩 레지스터 주소표 (0번부터). Modbus 레지스터는 16비트 정수만 저장하므로
# 소수는 배율을 곱해 정수로 저장함 (예: 380.5V → 3805)

REG = dict(
    voltage=0,    # 전압 V × 10
    current=1,    # 전류 A × 10
    kw=2,         # 유효전력 kW × 10 (계통에서 받는 전력)
    pf=3,         # 역률 × 1000
    hz=4,         # 주파수 Hz × 100
    kwh_hi=5,     # 누적 전력량 kWh × 10 의 상위 16비트
    kwh_lo=6,     # 누적 전력량 kWh × 10 의 하위 16비트 (둘을 합쳐 32비트)
    soc=10,       # ESS 충전율 % × 10
    ess_kw=11,    # ESS 실제 방전 출력 kW × 10
    ess_cmd=20,   # ESS 방전 지령 kW × 10 (제어기가 씀)
    scenario=30,  # 시나리오 번호 (scenario.py가 씀)
)




# 코일 = 켜짐/꺼짐 1비트 값. 부하 3개의 운전 상태 (1=운전, 0=차단)
LOADS = {"load1": 0, "load2": 1, "load3": 2}           # 부하 이름 → 코일 주소
LOAD_KW = {"load1": 150, "load2": 120, "load3": 100}   # 차단 가능한 부하 크기 (kW)
SHED_ORDER = ["load3", "load2", "load1"]               # 차단 순서: 중요도 낮은 것부터


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)