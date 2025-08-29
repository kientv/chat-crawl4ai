FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install only Chromium browser (not all browsers)
RUN playwright install chromium

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