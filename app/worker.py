"""Worker entrypoint: python -m app.worker"""
from app.jobs import worker_loop

if __name__ == "__main__":
    worker_loop()
