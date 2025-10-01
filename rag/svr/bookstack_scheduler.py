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

            # Get all BookStack documents directly
            bookstack_docs = self._get_bookstack_documents()

            for doc in bookstack_docs:
                if self._check_document_updates(doc):
                    self._schedule_document_update(doc)

        except Exception as e:
            logging.exception(f"Error checking BookStack updates: {str(e)}")


    def _get_bookstack_documents(self) -> List[Dict[str, Any]]:
        """Get all BookStack documents"""
        try:
            from api.db.services.document_service import DocumentService

            # Query trực tiếp documents có parser_id = "bookstack"
            bookstack_docs = list(DocumentService.model.select().where(
                DocumentService.model.parser_id == "bookstack"
            ).dicts())

            logging.info(f"Found {len(bookstack_docs)} BookStack documents")
            return bookstack_docs

        except Exception as e:
            logging.exception(f"Error getting BookStack documents: {str(e)}")
            return []

    def _check_document_updates(self, doc: Dict[str, Any]) -> bool:
        """Check if document has updates"""
        try:
            parser_config = doc.get("parser_config", {})
            source_chapter_id = parser_config.get("source_chapter_id")

            if not source_chapter_id:
                return False

            # Get BookStack config
            global_config = settings.BOOKSTACK_CONFIG or {}
            if not global_config:
                logging.warning("No BookStack config found")
                return False

            # Initialize connector
            connector = BookStackConnector(
                base_url=global_config["base_url"],
                token_id=global_config["token_id"],
                token_secret=global_config["token_secret"]
            )

            # Get chapter info from BookStack
            try:
                chapter_data = connector.client.get_chapter_content(source_chapter_id)
                bookstack_updated_at = datetime.fromisoformat(
                    str(chapter_data.get('updated_at', '')).replace('Z', '+00:00')
                )

                # Compare với updated_at của document
                doc_updated_at = doc.get("update_time")
                if doc_updated_at:
                    if isinstance(doc_updated_at, str):
                        doc_updated_at = datetime.fromisoformat(doc_updated_at)

                    # Nếu BookStack update sau document update time
                    if bookstack_updated_at > doc_updated_at:
                        logging.info(f"Document {doc['id']} has updates: BookStack={bookstack_updated_at}, Doc={doc_updated_at}")
                        return True

            except Exception as e:
                logging.warning(f"Error checking document {doc['id']} updates: {str(e)}")
                return False

            return False

        except Exception as e:
            logging.exception(f"Error checking document updates: {str(e)}")
            return False

    def _schedule_document_update(self, doc: Dict[str, Any]):
        """Schedule update task cho single document"""
        try:
            task_data = {
                "id": get_uuid(),
                "doc_id": doc["id"],
                "kb_id": doc["kb_id"],
                "tenant_id": doc["tenant_id"],
                "name": doc["name"],
                "parser_id": doc["parser_id"],
                "parser_config": doc["parser_config"],
                "from_page": 0,
                "to_page": -1,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "progress": 0.0,
                "progress_msg": "Scheduled for BookStack sync",
                "created_by": "bookstack_scheduler"
            }

            # Insert task
            TaskService.insert(task_data)

            # Queue task for page-level sync
            REDIS_CONN.queue_product("task_executor", {
                "id": task_data["id"],
                "task_type": "bookstack"  # For page updates
            })

            logging.info(f"Scheduled update task for document {doc['id']}: {doc['name']}")

        except Exception as e:
            logging.exception(f"Error scheduling document update: {str(e)}")

    def _schedule_document_updates(self, docs: List[Dict[str, Any]]):
        """Schedule update tasks cho documents"""
        try:
            for doc in docs:
                task_data = {
                    "id": get_uuid(),
                    "doc_id": doc["id"],
                    "kb_id": doc["kb_id"],
                    "tenant_id": doc["tenant_id"],
                    "name": doc["name"],
                    "parser_id": doc["parser_id"],
                    "parser_config": doc["parser_config"],
                    "from_page": 0,
                    "to_page": -1,
                    "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "progress": 0.0,
                    "progress_msg": "Scheduled for BookStack sync",
                    "created_by": "bookstack_scheduler"
                }

                # Insert task
                TaskService.insert(task_data)

                # Queue task
                REDIS_CONN.queue_product("task_executor", {
                    "id": task_data["id"],
                    "task_type": "chunk"
                })

                logging.info(f"Scheduled update task for document {doc['id']}: {doc['name']}")

        except Exception as e:
            logging.exception(f"Error scheduling document updates: {str(e)}")

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