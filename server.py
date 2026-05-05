import os
import threading
import time
import schedule
from waitress import serve
from app import app
from train_job import run_train_job

def run_scheduler():
    """Background thread to regularly run scheduled jobs."""
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    
    schedule.every().day.at("00:00").do(run_train_job)
    
    schedule.every(12).hours.do(run_train_job)
    
    print(f"Starting scheduled tasks. Next training scheduled at: {schedule.next_run()}")
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    print(f"Starting Waitress server on port {port}...")
    serve(app, host='0.0.0.0', port=port)
