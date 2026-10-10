# 파이썬 3.12가 깔린 가벼운 리눅스 이미지에서 시작
FROM python:3.12-slim

# 컨테이너 안 작업 폴더
WORKDIR /app

# 패키지 목록을 먼저 복사해 설치 (코드만 바뀌면 이 단계는 캐시로 건너뜀)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest.py .

# print가 바로바로 로그에 찍히게 (버퍼링 끄기)
ENV PYTHONUNBUFFERED=1

# 컨테이너가 켜지면 실행할 명령
CMD ["python", "ingest.py"]