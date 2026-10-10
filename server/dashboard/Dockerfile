FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
# 0.0.0.0: 컨테이너 밖(브라우저, 터널)에서 접속할 수 있게
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]