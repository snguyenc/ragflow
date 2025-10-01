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
from api.db.services.task_service import queue_tasks, incremental_queue_tasks
from api.db.services.file2document_service import File2DocumentService
from api.db.services.task_service import TaskService

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

        has, doc = DocumentService.get_by_id(doc_id)
        if not has:
            progress_callback(-1, f"Document {doc_id} not found")
            return []

        progress_callback(0.2, f"Initializing BookStack connector for books: {', '.join(booknames)}")
        doc = doc.to_dict()
        #print("doc: ", doc)
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

        connector_page = BookStackConnector(
            base_url=global_config["base_url"],
            token_id=global_config["token_id"],
            token_secret=global_config["token_secret"],
            batch_size=50,
            include_pages=True  # Primary method for book filtering
        )

        progress_callback(0.3, "Testing BookStack connection...")
        success, error = connector.test_connection()
        if not success:
            raise Exception(f"BookStack connection failed: {error}")

        progress_callback(0.4, "Fetching chapters from BookStack...")

        # Fetch chapters and pages for specified books
        # not usfull any more, chunk num alway reset
        doc_count = 0
        doc_chunk_num = doc.get("chunk_num")
        doc_updated_at = doc.get("update_date") if doc_chunk_num > 0 else None

        logging.info(f"Fetching chapters from BookStack for books: {doc_updated_at}, {doc_chunk_num}")
        
        for batch in connector.fetch_documents(book_names=booknames, updated_since=doc_updated_at):
            for bookstack_doc in batch:
                doc_count += 1
                progress_callback(0.4 + 0.5 * doc_count / 100, f"Processing {bookstack_doc.doc_type}: {bookstack_doc.title}")
                category = bookstack_doc.metadata.get("book_name", "")
                parser_config = {**task_parser_config, "source_chapter_id": bookstack_doc.doc_id,
                 "category": category, "guide": bookstack_doc.title}
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
                    chapter_docs = chapter_document.to_dict()
                    #keep updated_at for run job
                    from datetime import timezone
                    parser_config["updated_at"] = chapter_docs.get("update_date").astimezone(tz=timezone.utc).isoformat()
                    DocumentService.update_parser_config(chapter_document.id, parser_config)
                    logging.info(f"Chapter document {bookstack_doc.title} already exists, using existing...")

                    page_count = 0
                    for batch in connector_page.fetch_documents(chapter_id=bookstack_doc.doc_id, updated_since=chapter_docs.get("update_date")):
                        for bookstack_doc in batch:
                            page_count += 1
                            
                    if page_count > 0:
                        logging.info(f"Submit task for auto chunking: {chapter_docs['id']}")
                        bucket, name = File2DocumentService.get_storage_address(doc_id=chapter_docs['id'])
                        incremental_queue_tasks(chapter_docs, bucket, name, 1)
                    else: 
                        logging.info(f"No new pages found for chapter: {bookstack_doc.title}")    
                else:
                    # Insert new chapter document
                    chapter_document = DocumentService.insert(chapter_doc_data)
                    chapter_docs = chapter_document.to_dict()
                    # Add to file system
                    kb_folder = FileService.get_kb_folder(tenant_id)
                    FileService.add_file_from_kb(chapter_docs, kb_folder["id"], tenant_id)
                    logging.info(f"Created chapter document: {chapter_document.id} for chapter: {bookstack_doc.title}")

                    #submit task for auto chunking
                    logging.info(f"Submit task for auto chunking: {chapter_docs['id']}")
                    bucket, name = File2DocumentService.get_storage_address(doc_id=chapter_docs['id'])
                    queue_tasks(chapter_docs, bucket, name, 1)

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
        tenant_id = kwargs.get("tenant_id")
        kb_id = kwargs.get("kb_id")
        doc_id = kwargs.get("doc_id")

        chapter_id = parser_config.get("source_chapter_id")
        updated_at_str = parser_config.get("updated_at")

        if not chapter_id:
            raise ValueError("doc_id not found in parser_config")
        has, doc = DocumentService.get_by_id(doc_id)
        if not has:
            callback(-1, f"Document {doc_id} not found")
            return []
        
        doc = doc.to_dict()    
        prev_tasks = TaskService.get_tasks(doc_id)
        pre_chunk_ids = []
        if prev_tasks:
            for pre_task in prev_tasks:
                pre_chunk_ids.extend(pre_task["chunk_ids"].split())
            
        
        updated_at = datetime.fromisoformat(updated_at_str) if len(pre_chunk_ids) > 0 and updated_at_str else None

        doc_count = 0
        docs = []

        
        for batch in connector.fetch_documents(chapter_id=chapter_id, updated_since=updated_at):
            for bookstack_doc in batch:
                doc_count += 1
                callback(0.4 + 0.5 * doc_count / 100, f"Processing {bookstack_doc.doc_type}: {bookstack_doc.title}")
                bookstack_doc.metadata = {
                    **bookstack_doc.metadata,
                    **parser_config
                }
                docs.append(bookstack_doc)
                # check and delete old chunk
                if settings.docStoreConn.indexExist(search.index_name(tenant_id), kb_id):
                    logging.info(f"Delete old chunk: {bookstack_doc.metadata['chapter_id']}, {bookstack_doc.metadata['page_id']}")
                    settings.docStoreConn.delete({"doc_id": doc_id, "chapter_id": bookstack_doc.metadata['chapter_id'], "page_id": bookstack_doc.metadata['page_id']}, 
                    search.index_name(tenant_id), kb_id)

        logging.info(f"Fetching pages from BookStack: {updated_at}, {docs}")
        return docs

    except Exception as e:
        callback(-1, f"BookStack chapter fetch error: {str(e)}")
        logging.error(f"BookStack chapter fetch error: {str(e)}")
        raise

