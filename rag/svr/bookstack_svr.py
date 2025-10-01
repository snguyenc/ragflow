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


async def run_bookstack_connector(task, embedding_model, progress_callback):
    """
    Run BookStack connector to fetch documents and convert to chunks
    """
    try:
        # Get config from task
        doc_config = task.get("parser_config", {}).get("bookstack", {})
        doc_id = task.get("doc_id")
        kb_id = task["kb_id"]
        tenant_id = task["tenant_id"]

        # Get booknames filter
        booknames_filter = doc_config.get("booknames", [])

        # Use standalone function
        return await fetch_bookstack_content(
            bookstack_config=doc_config,
            kb_id=kb_id,
            tenant_id=tenant_id,
            progress_callback=progress_callback,
            doc_id=doc_id,
            booknames_filter=booknames_filter
        )

    except Exception as e:
        progress_callback(-1, f"BookStack connector failed: {str(e)}")
        logging.exception("BookStack connector error")
        raise


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

                # Create document for this chapter
                chapter_doc_data = {
                    "id": get_uuid(),
                    "kb_id": kb_id,
                    "parser_id": "bookstack",
                    "parser_config": {"source_chapter_id": bookstack_doc.doc_id, "booknames": booknames},
                    "created_by": "task_executor",
                    "type": FileType.DOC,
                    "name": f"Chapter: {bookstack_doc.title}",
                    "suffix": "chapter",
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


async def fetch_bookstack_content(bookstack_config, kb_id, tenant_id, progress_callback, doc_id=None, booknames_filter=None, return_raw=False):
    """
    Fetch content from BookStack - can be used standalone

    Args:
        bookstack_config: BookStack configuration
        kb_id: Knowledge base ID
        tenant_id: Tenant ID
        progress_callback: Progress callback function
        doc_id: Optional document ID to associate chunks with
        booknames_filter: Optional list of book names to filter by

    Returns:
        List of processed chunks from BookStack
    """

    try:
        progress_callback(0.1, "Initializing BookStack connector...")

        # Get BookStack configuration from settings and merge with provided config
        from api import settings
        global_config = settings.BOOKSTACK_CONFIG or {}
        merged_config = {**global_config, **bookstack_config}

        if not merged_config:
            raise ValueError("BookStack configuration not found")

        base_url = merged_config.get("base_url")
        token_id = merged_config.get("token_id")
        token_secret = merged_config.get("token_secret")

        if not all([base_url, token_id, token_secret]):
            raise ValueError("Missing required BookStack credentials")

        # Initialize connector
        connector = BookStackConnector(
            base_url=base_url,
            token_id=token_id,
            token_secret=token_secret,
            batch_size=merged_config.get("batch_size", 50),
            include_books=merged_config.get("include_books", True),
            include_chapters=merged_config.get("include_chapters", True),
            include_pages=merged_config.get("include_pages", True),
            include_shelves=merged_config.get("include_shelves", True)
        )

        # Test connection
        progress_callback(0.2, "Testing BookStack connection...")
        success, error = connector.test_connection()
        if not success:
            raise Exception(f"BookStack connection failed: {error}")

        progress_callback(0.3, "Fetching documents from BookStack...")

        # Parse date filters if provided
        updated_since = None
        updated_until = None
        if merged_config.get("updated_since"):
            try:
                from datetime import datetime
                updated_since = datetime.fromisoformat(merged_config["updated_since"])
            except ValueError:
                logging.warning(f"Invalid updated_since date format: {merged_config['updated_since']}")

        if merged_config.get("updated_until"):
            try:
                from datetime import datetime
                updated_until = datetime.fromisoformat(merged_config["updated_until"])
            except ValueError:
                logging.warning(f"Invalid updated_until date format: {merged_config['updated_until']}")

        all_chunks = []
        total_documents = 0

        def fetch_progress(prog, msg):
            progress_callback(0.3 + 0.5 * prog, msg)

        for document_batch in connector.fetch_documents(
            updated_since=updated_since,
            updated_until=updated_until,
            progress_callback=fetch_progress
        ):
            # Convert documents to RAGFlow chunks
            for document in document_batch:

                chunk = document.to_ragflow_chunk(
                    kb_id=kb_id,
                    tenant_id=tenant_id
                )

                # Use provided doc_id or generate virtual doc_id
                if doc_id:
                    chunk["doc_id"] = doc_id
                else:
                    chunk["doc_id"] = f"bookstack_{document.doc_type}_{document.metadata.get('bookstack_id', 'unknown')}"

                all_chunks.append(chunk)
                total_documents += 1

        progress_callback(0.8, f"Processed {total_documents} documents from BookStack")

        # Process chunks similar to standard chunking
        res_chunks = []
        tk_count = 0
        for chunk in all_chunks:
            content = chunk["content_with_weight"]
            if content.strip():
                chunk["content_ltks"] = rag_tokenizer.tokenize(content)
                chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
                res_chunks.append(chunk)
                tk_count += num_tokens_from_string(content)

        progress_callback(1.0, f"BookStack fetch completed. {len(res_chunks)} chunks, {tk_count} tokens")
        return res_chunks

    except Exception as e:
        progress_callback(-1, f"BookStack fetch failed: {str(e)}")
        logging.exception("BookStack fetch error")
        raise


