FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DB_PATH=/data/predictions.db
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

RUN useradd --create-home appuser && mkdir /data && chown appuser /data
USER appuser
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# ONE worker on purpose: the scheduler lives inside the process, and more workers would
# each run their own copy (duplicate runs and alerts). Scale reads separately if needed.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
