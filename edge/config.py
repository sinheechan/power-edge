# 엣지 공통 설정 (simulator, controller, scenario 가 같이 씀)
# 비밀번호만 .env 에서 읽고 나머지는 여기에 직접 적음

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")

MODBUS_HOST = "localhost"
MODBUS_PORT = 5020
MQTT_HOST = "localhost"
MQTT_PORT = 1883
SITE = "plant1"

TARGET_KW = 1600   # 목표 최대수요전력 (15분 평균)
ESS_KW = 200       # ESS 방전 출력

# 차단할 수 있는 부하 3개 (순서 = 코일 주소 0, 1, 2 = load1, load2, load3)
LOAD_KW = [150, 120, 100]
SHED_ORDER = [2, 1, 0]   # 덜 중요한 load3 부터 차단

# 레지스터 주소표. Modbus 는 정수만 저장해서 소수는 배율을 곱해 넣음 (380.5V -> 3805)
#   0: 전압 x10      1: 전류 x10      2: kW x10       3: 역률 x1000    4: Hz x100
#   5, 6: 누적 kWh x10 (32비트를 상위/하위 16비트로 나눔)
#   10: ESS 충전율 x10    11: ESS 방전 kW x10
#   20: ESS 방전 지령 kW x10 (controller 가 씀)
#   30: 시나리오 번호 (scenario.py 가 씀)


def now_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
