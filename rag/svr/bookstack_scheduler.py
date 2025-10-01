#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any

from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import TaskService
from api.utils import get_uuid
from api import settings
from rag.connectors.bookstack_connector import BookStackConnector
from rag.utils.redis_conn import REDIS_CONN
from api.db.services.file2document_service import File2DocumentService
from api.db.services.task_service import queue_tasks
from api.db import FileType
class BookStackScheduler:
    """
    Scheduler để sync BookStack content định kỳ
    """

    def __init__(self, check_interval_minutes: int = 60):
        """
        Args:
            check_interval_minutes: Khoảng thời gian check (phút)
        """
        self.check_interval = check_interval_minutes
        self.is_running = False
        self.scheduler_thread = None

    def start(self):
        """Start scheduler"""
        if self.is_running:
            logging.warning("BookStack scheduler is already running")
            return

        logging.info(f"Starting BookStack scheduler with {self.check_interval} minutes interval")

        # Schedule job
        schedule.every(self.check_interval).minutes.do(self._check_bookstack_updates)

        # Chạy ngay lần đầu
        #schedule.every().day.at("00:00").do(self._daily_full_sync)

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

    def stop(self):
        """Stop scheduler"""
        self.is_running = False
        schedule.clear()
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logging.info("BookStack scheduler stopped")

    def _run_scheduler(self):
        """Run scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _check_bookstack_updates(self):
        """Check BookStack for updates"""
        try:
            logging.info("Checking BookStack for updates...")
            existing_docs = list(DocumentService.model.select().where(
                DocumentService.model.type == FileType.BOOKSTACK,
                DocumentService.model.parser_id == "bookstack_chapter_doc"
            ))

            if existing_docs:
                for doc in existing_docs:
                    doc_dict = doc.to_dict()
                    parser_config = doc_dict.get("parser_config")
                    from datetime import timezone
                    parser_config["updated_at"] = doc_dict.get("update_date").astimezone(tz=timezone.utc).isoformat()
                    
                    DocumentService.update_parser_config(doc.id, parser_config)
                    bucket, name = File2DocumentService.get_storage_address(doc_id=doc_dict["id"])
                    queue_tasks(doc_dict, bucket, name, 0)

            

        except Exception as e:
            logging.exception(f"Error checking BookStack updates: {str(e)}")




# Global scheduler instance
bookstack_scheduler = BookStackScheduler()


def start_bookstack_scheduler():
    """Start BookStack scheduler"""
    from api import settings

    # Check if scheduler is enabled
    if not settings.BOOKSTACK_CONFIG:
        logging.info("BookStack config not found, scheduler disabled")
        return

    scheduler_config = settings.BOOKSTACK_CONFIG.get("scheduler", {})
    if not scheduler_config.get("enabled", True):
        logging.info("BookStack scheduler disabled in config")
        return

    # Get interval from config
    check_interval = scheduler_config.get("check_interval_minutes", 60)

    # Update scheduler interval
    global bookstack_scheduler
    bookstack_scheduler = BookStackScheduler(check_interval_minutes=check_interval)
    bookstack_scheduler.start()


def stop_bookstack_scheduler():
    """Stop BookStack scheduler"""
    bookstack_scheduler.stop()


if __name__ == "__main__":
    # Test scheduler
    logging.basicConfig(level=logging.INFO)
    scheduler = BookStackScheduler(check_interval_minutes=1)  # 1 phút cho test
    scheduler.start()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        scheduler.stop()