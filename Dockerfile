# Đổi base image cho nhẹ hơn, các phần còn lại giữ nguyên
FROM python:3.12-slim

# Cài libs tối thiểu để Chromium (Playwright) chạy được
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget \
    libglib2.0-0 libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 \
    libgtk-3-0 libgbm1 libasound2 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install only Chromium browser (not all browsers)
RUN pip install --no-cache-dir playwright && \
    playwright install chromium

# Setup tiktoken cache for offline usage
RUN mkdir -p /opt/tiktoken_cache
COPY 9b5ad71b2ce5302211f9c61530b329a4922fc6a4 /opt/tiktoken_cache/
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# Copy environment file and Python scripts
COPY .env ./
COPY *.py ./

EXPOSE 8881

# Use main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8881"]
