from api.db.services.document_service import DocumentService
from api.db import FileType
from api.utils import get_uuid
from rag.nlp import rag_tokenizer
import settings
from rag.nlp import search
from rag.connectors.bookstack_connector import BookStackConnector
import logging
from datetime import datetime



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
            include_book_to_chapters=True
        )

        progress_callback(0.3, "Testing BookStack connection...")
        success, error = connector.test_connection()
        if not success:
            raise Exception(f"BookStack connection failed: {error}")

        progress_callback(0.4, "Fetching chapters from BookStack...")

        # Fetch chapters for specified books
        all_chunks = []
        chapter_count = 0

        for batch in connector.fetch_documents(book_names=booknames):
            for chapter_doc in batch:
                chapter_count += 1
                progress_callback(0.4 + 0.4 * chapter_count / 100, f"Processing chapter: {chapter_doc.title}")

                # Create document for this chapter
                chapter_doc_data = {
                    "id": get_uuid(),
                    "kb_id": kb_id,
                    "parser_id": "bookstack",
                    "parser_config": {"chapter_id": chapter_doc.doc_id, "parent_id": doc_id},
                    "created_by": "task_executor",
                    "type": FileType.DOC,
                    "name": chapter_doc.title,
                    "suffix": "chapter",
                    "location": f"{doc_id}-{chapter_doc.doc_id}",
                    "size": len(chapter_doc.content.encode('utf-8')) if chapter_doc.content else 0,
                }

                existing_docs = list(DocumentService.model.select().where(
                    DocumentService.model.kb_id == kb["id"],
                    DocumentService.model.type == FileType.DOC,
                    DocumentService.model.parser_id == "bookstack",
                    DocumentService.model.location == f"{doc_id}-{chapter_doc.doc_id}"
                ))

                if existing_docs:
                    logging.info(f"Chapter document {chapter_doc.title} already exists, skipping...")
                    info = {"run": str(1), "progress": 0, "progress_msg": "", "chunk_num": 0, "token_num": 0}
                    DocumentService.update_by_id(existing_docs[0].id, info)
                else:
                    # Insert chapter document
                    chapter_document = DocumentService.insert(chapter_doc_data)
                    logging.info(f"Created chapter document: {chapter_document.id} for chapter: {chapter_doc.title}")

                # Get pages for this chapter
                progress_callback(0.4 + 0.4 * chapter_count / 100, f"Fetching pages for chapter: {chapter_doc.title}")

                try:
                    # Get chapter pages
                    chapter_id = chapter_doc.doc_id
                    pages = connector.client.get_pages_from_chapter(chapter_id)

                    # Create chunks from pages
                    page_chunks = []
                    for page in pages:
                        if page.get('html') or page.get('markdown'):
                            content = page.get('markdown', '') or page.get('html', '')

                            # Create chunk for this page
                            chunk = {
                                "id": get_uuid(),
                                "doc_id": chapter_document.id,
                                "kb_id": [kb_id],
                                "docnm_kwd": chapter_document.name,
                                "title_tks": rag_tokenizer.tokenize(page.get('name', '')),
                                "content_ltks": rag_tokenizer.tokenize(content),
                                "content_with_weight": content,
                                "page_num": page.get('priority', 0),
                                "source_type": "bookstack_page",
                                "source_id": str(page.get('id', '')),
                                "create_time": str(datetime.now()).replace("T", " ")[:19],
                                "create_timestamp_flt": datetime.now().timestamp()
                            }
                            page_chunks.append(chunk)

                    # Add chunks to collection
                    all_chunks.extend(page_chunks)
                    logging.info(f"Created {len(page_chunks)} chunks from {len(pages)} pages for chapter: {chapter_doc.title}")

                except Exception as page_error:
                    logging.warning(f"Failed to get pages for chapter {chapter_doc.title}: {str(page_error)}")
                    # Create single chunk from chapter content if pages fail
                    if chapter_doc.content:
                        chunk = {
                            "id": get_uuid(),
                            "doc_id": chapter_document.id,
                            "kb_id": [kb_id],
                            "docnm_kwd": chapter_document.name,
                            "title_tks": rag_tokenizer.tokenize(chapter_doc.title),
                            "content_ltks": rag_tokenizer.tokenize(chapter_doc.content),
                            "content_with_weight": chapter_doc.content,
                            "page_num": 0,
                            "source_type": "bookstack_chapter",
                            "source_id": chapter_doc.doc_id,
                            "create_time": str(datetime.now()).replace("T", " ")[:19],
                            "create_timestamp_flt": datetime.now().timestamp()
                        }
                        all_chunks.append(chunk)

        progress_callback(0.8, f"Created {chapter_count} chapter documents with {len(all_chunks)} total chunks")

        # Store chunks in document store
        if all_chunks:
            progress_callback(0.9, "Storing chunks in document store...")
            idxnm = search.index_name(tenant_id)

            # Ensure index exists
            if not settings.docStoreConn.indexExist(idxnm, kb_id):
                settings.docStoreConn.createIdx(idxnm, kb_id, 1536)  # Default vector size

            # Insert chunks in batches
            batch_size = 64
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                settings.docStoreConn.insert(batch, idxnm, kb_id)

        progress_callback(1.0, f"BookStack sync complete: {chapter_count} chapters, {len(all_chunks)} chunks")
        return all_chunks

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


