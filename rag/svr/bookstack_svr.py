from api.db.services.document_service import DocumentService
from api.db import FileType
from api.utils import get_uuid
from rag.nlp import rag_tokenizer
from api import settings
from rag.nlp import search
from rag.connectors.bookstack_connector import BookStackConnector
import logging
from datetime import datetime
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.file_service import FileService

async def run_bookstack_chapter_doc(task, progress_callback):
    """
    Run BookStack connector to fetch chapters and create documents
    - Fetch chapters based on book names
    - Each chapter becomes a document (type = doc)
    - Each page in chapter becomes a chunk in that document
    """
    try:
      

        # Get task info
        doc_id = task["doc_id"]
        kb_id = task.get("kb_id")
        tenant_id = task.get("tenant_id")
        task_parser_config = task["parser_config"]

        booknames = task_parser_config.get("booknames", [])
        if not booknames:
            progress_callback(-1, "No booknames provided in document parser_config")
            return []

        progress_callback(0.2, f"Initializing BookStack connector for books: {', '.join(booknames)}")

        # Get BookStack config from settings
        global_config = settings.BOOKSTACK_CONFIG or {}
        if not global_config or not all([
            global_config.get("base_url"),
            global_config.get("token_id"),
            global_config.get("token_secret")
        ]):
            raise ValueError("Missing BookStack configuration in settings")

        connector = BookStackConnector(
            base_url=global_config["base_url"],
            token_id=global_config["token_id"],
            token_secret=global_config["token_secret"],
            batch_size=50,
            include_book_to_chapters=True  # Primary method for book filtering
        )

        progress_callback(0.3, "Testing BookStack connection...")
        success, error = connector.test_connection()
        if not success:
            raise Exception(f"BookStack connection failed: {error}")

        progress_callback(0.4, "Fetching chapters from BookStack...")

        # Fetch chapters and pages for specified books
        doc_count = 0

        for batch in connector.fetch_documents(book_names=booknames):
            for bookstack_doc in batch:
                doc_count += 1
                progress_callback(0.4 + 0.5 * doc_count / 100, f"Processing {bookstack_doc.doc_type}: {bookstack_doc.title}")
                category = bookstack_doc.metadata.get("book_name", "")
                parser_config = {**task_parser_config, "source_chapter_id": bookstack_doc.doc_id, "category": category, "guide": bookstack_doc.title}
                # Create document for this chapter
                chapter_doc_data = {
                    "id": get_uuid(),
                    "kb_id": kb_id,
                    "parser_id": "bookstack",
                    "parser_config": parser_config,
                    "created_by": "task_executor",
                    "type": FileType.DOC,
                    "name": f"{bookstack_doc.title}.bookstack",
                    "suffix": "bookstack",
                    "location": bookstack_doc.url,
                    "metafields": {
                        "created_at": bookstack_doc.created_at,
                        "updated_at": bookstack_doc.updated_at,
                        **bookstack_doc.metadata,
                    },
                    "size": len(bookstack_doc.content.encode('utf-8')) if bookstack_doc.content else 0,
                }

                # Check if chapter document already exists
                existing_docs = list(DocumentService.model.select().where(
                    DocumentService.model.kb_id == kb_id,
                    DocumentService.model.type == FileType.DOC,
                    DocumentService.model.parser_id == "bookstack",
                    DocumentService.model.location == bookstack_doc.url
                ))

                if existing_docs:
                    chapter_document = existing_docs[0]
                    DocumentService.update_parser_config(chapter_document.id, parser_config)
                    logging.info(f"Chapter document {bookstack_doc.title} already exists, using existing...")
                else:
                    # Insert new chapter document
                    chapter_document = DocumentService.insert(chapter_doc_data)
                    # Add to file system
                    kb_folder = FileService.get_kb_folder(tenant_id)
                    FileService.add_file_from_kb(chapter_document.to_dict(), kb_folder["id"], tenant_id)
                    logging.info(f"Created chapter document: {chapter_document.id} for chapter: {bookstack_doc.title}")

        return doc_count

    except Exception as e:
        progress_callback(-1, f"BookStack chapter fetch failed: {str(e)}")
        logging.exception("BookStack chapter fetch error")
        raise



def fetch_bookstack_pages(parser_config, kwargs, callback):
    """
    Fetch BookStack chapter content and convert pages to sections format

    Args:
        parser_config: Parser configuration containing source_chapter_id
        kwargs: Additional arguments containing tenant_id, etc.
        callback: Progress callback function

    Returns:
        List of sections in format [(content, ""), ...]
    """
    try:

        # Get BookStack configuration
        global_config = settings.BOOKSTACK_CONFIG or {}
        if not global_config or not all([
            global_config.get("base_url"),
            global_config.get("token_id"),
            global_config.get("token_secret")
        ]):
            raise ValueError("Missing BookStack configuration in settings")

        # Initialize connector
        connector = BookStackConnector(
            base_url=global_config["base_url"],
            token_id=global_config["token_id"],
            token_secret=global_config["token_secret"],
            batch_size=50,
            include_chapter_to_pages=True
        )

        callback(0.2, "Testing BookStack connection...")
        success, error = connector.test_connection()
        if not success:
            raise Exception(f"BookStack connection failed: {error}")

        callback(0.3, "Fetching chapter pages from BookStack...")

        # Get chapter ID from parser config
        chapter_id = parser_config.get("source_chapter_id")
        if not chapter_id:
            raise ValueError("source_chapter_id not found in parser_config")

        doc_count = 0
        docs = []
        for batch in connector.fetch_documents(chapter_id=chapter_id):
            for bookstack_doc in batch:
                doc_count += 1
                callback(0.4 + 0.5 * doc_count / 100, f"Processing {bookstack_doc.doc_type}: {bookstack_doc.title}")
                bookstack_doc.metadata = {
                    **bookstack_doc.metadata,
                    **parser_config
                }
                docs.append(bookstack_doc)
        
        return docs

    except Exception as e:
        callback(-1, f"BookStack chapter fetch error: {str(e)}")
        logging.error(f"BookStack chapter fetch error: {str(e)}")
        raise

