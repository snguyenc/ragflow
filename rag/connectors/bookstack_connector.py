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

import html
import json
import logging
import re
import time
import xxhash
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple
from io import BytesIO
from functools import partial
from rag.connectors.bookstack_client import BookStackClient, BookStackClientError


class BookStackDocument:
    """
    Document representation for BookStack content

    Provides unified interface for different BookStack content types (books, chapters, pages, shelves)
    """

    def __init__(self,
                 doc_id: str,
                 title: str,
                 content: str,
                 url: str,
                 doc_type: str,
                 updated_at: Optional[datetime] = None,
                 created_at: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.url = url
        self.doc_type = doc_type
        self.updated_at = updated_at
        self.created_at = created_at
        self.metadata = metadata or {}

    def to_ragflow_chunk(self, kb_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Convert BookStack document to RAGFlow chunk format

        Args:
            kb_id: Knowledge base ID
            tenant_id: Tenant ID

        Returns:
            RAGFlow chunk data structure
        """
        # Generate unique ID based on content and doc_id
        content_hash = xxhash.xxh64((self.content + self.doc_id).encode("utf-8", "surrogatepass")).hexdigest()

        chunk = {
            "id": content_hash,
            "content_with_weight": self.content,
            "doc_id": self.doc_id,
            "docnm_kwd": self.title,
            "title_tks": self.title.lower(),
            "kb_id": str(kb_id),
            "tenant_id": str(tenant_id),
            "create_time": str(datetime.now()).replace("T", " ")[:19],
            "create_timestamp_flt": datetime.now().timestamp(),
            "img_id": "",
            "page_num_int": [1],
            "position_int": [0, len(self.content)],
            "top_int": [0],
            "available_int": 1,
            "metadata": {
                "source": "bookstack",
                "type": self.doc_type,
                "url": self.url,
                #"updated_at": self.updated_at.isoformat() if self.updated_at else None,
                **self.metadata
            }
        }

        return chunk


class BookStackConnector:
    """
    BookStack Connector for RAGFlow

    Fetches content from BookStack knowledge management system and converts it
    to RAGFlow-compatible document chunks for indexing.
    """

    def __init__(self,
                 base_url: str,
                 token_id: str,
                 token_secret: str,
                 batch_size: int = 50,
                 include_books: bool = False,
                 include_chapters: bool = False,
                 include_pages: bool = False,
                 include_shelves: bool = False,
                include_book_to_chapters: bool = False,
                include_chapter_to_pages: bool = False):
        """
        Initialize BookStack connector

        Args:
            base_url: BookStack instance URL
            token_id: API token ID
            token_secret: API token secret
            batch_size: Batch size for API requests
            include_books: Whether to include books
            include_chapters: Whether to include chapters
            include_pages: Whether to include pages
            include_shelves: Whether to include shelves
            include_book_to_chapters: Whether to include book to chapters
            include_chapter_to_pages: Whether to include chapter to pages
        """
        self.client = BookStackClient(base_url, token_id, token_secret)
        self.batch_size = batch_size
        self.include_books = include_books
        self.include_chapters = include_chapters
        self.include_pages = include_pages
        self.include_shelves = include_shelves
        self.include_book_to_chapters = include_book_to_chapters
        self.include_chapter_to_pages = include_chapter_to_pages

    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test connection to BookStack

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not self.client.test_connection():
                return False, "Unable to connect to BookStack API"
            return True, None
        except BookStackClientError as e:
            return False, f"BookStack connection error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML content to plain text

        Args:
            html_content: HTML content string

        Returns:
            Cleaned plain text
        """
        if not html_content:
            return ""

        # Unescape HTML entities
        text = html.unescape(html_content)

        # Remove script and style elements
        text = re.sub(r'<(script|style).*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags but keep the content
        text = re.sub(r'<[^>]+>', ' ', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _book_to_document(self, book_data: Dict[str, Any]) -> BookStackDocument:
        """Convert BookStack book to document"""
        book_id = str(book_data.get('id', ''))
        title = str(book_data.get('name', 'Untitled Book'))
        description = str(book_data.get('description', ''))

        content = f"{title}\n\n{description}"
        url = self.client.build_app_url(f"/books/{book_data.get('slug', book_id)}")

        updated_at, created_at = None, None
        if book_data.get('updated_at'):
            try:
                updated_at = datetime.fromisoformat(str(book_data['updated_at']).replace('Z', '+00:00'))
            except ValueError:
                pass
        if book_data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(str(book_data['created_at']).replace('Z', '+00:00'))
            except ValueError:
                pass

        return BookStackDocument(
            doc_id=f"bookstack_book_{book_id}",
            title=title,
            content=content,
            url=url,
            doc_type="book",
            updated_at=updated_at,
            created_at=created_at,
            metadata={
                "bookstack_id": book_id,
                "description": description
            }
        )

    def _chapter_to_document(self, chapter_data: Dict[str, Any]) -> BookStackDocument:
        """Convert BookStack chapter to document"""
        chapter_id = str(chapter_data.get('id', ''))
        title = str(chapter_data.get('name', ''))
        description = str(chapter_data.get('description', ''))

        content = f"{title}\n\n{description}"
        url = self.client.build_app_url(
            f"/books/{chapter_data.get('book_slug', '')}/chapter/{chapter_data.get('slug', chapter_id)}")

        updated_at = self._get_date(chapter_data.get('updated_at'))
        created_at = self._get_date(chapter_data.get('created_at'))

        return BookStackDocument(
            doc_id=chapter_id,
            title=title,
            content=content,
            url=url,
            doc_type="chapter",
            updated_at=updated_at,
            created_at=created_at,
            metadata={
                "chapter_id": chapter_id,
                "book_id": str(chapter_data.get('book_id', '')),
                "book_name": str(chapter_data.get('book_name', '')),
                "description": description
            }
        )

    def _shelf_to_document(self, shelf_data: Dict[str, Any]) -> BookStackDocument:
        """Convert BookStack shelf to document"""
        shelf_id = str(shelf_data.get('id', ''))
        title = str(shelf_data.get('name', 'Untitled Shelf'))
        description = str(shelf_data.get('description', ''))

        content = f"{title}\n\n{description}"
        url = self.client.build_app_url(f"/shelves/{shelf_data.get('slug', shelf_id)}")

        updated_at, created_at = None, None
        if shelf_data.get('updated_at'):
            try:
                updated_at = datetime.fromisoformat(str(shelf_data['updated_at']).replace('Z', '+00:00'))
            except ValueError:
                pass
        if shelf_data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(str(shelf_data['created_at']).replace('Z', '+00:00'))
            except ValueError:
                pass

        return BookStackDocument(
            doc_id=f"bookstack_shelf_{shelf_id}",
            title=title,
            content=content,
            url=url,
            doc_type="shelf",
            updated_at=updated_at,
            created_at=created_at,
            metadata={
                "bookstack_id": shelf_id,
                "description": description
            }
        )

    def _get_date(self, date_str: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
        except ValueError:
            return None
    
    def _page_to_document(self, page_data: Dict[str, Any]) -> BookStackDocument:
        """Convert BookStack page to document"""
        page_id = str(page_data.get('id', ''))
        title = str(page_data.get('name', ''))

        # Fetch detailed page content
        try:
            detailed_page = self.client.get_page_content(page_id)
            content_type = detailed_page.get('editor', 'markdown')
            md_content = detailed_page.get('markdown', '')
            if not md_content:
                md_content = detailed_page.get('html', '')

            revision_count = detailed_page.get('revision_count', 1)
            tags = detailed_page.get('tags', [])

            url = self.client.build_app_url(
            f"/books/{page_data.get('book_slug', '')}/page/{page_data.get('slug', page_id)}")

            updated_at = self._get_date(page_data.get('updated_at'))    
            created_at = self._get_date(page_data.get('created_at'))

            return BookStackDocument(
                doc_id=page_id,
                title=title,
                content=f"{title}\n\n{md_content}",
                url=url,
                doc_type=content_type,
                updated_at=updated_at,
                created_at=created_at,
                metadata={
                    "page_id": page_id,
                    "url": url,
                    "revision_count": revision_count,
                    "tags": tags,
                    "book_id": str(page_data.get('book_id', '')),
                    "chapter_id": str(page_data.get('chapter_id', '')),
                }
            )
        except BookStackClientError as e:
            logging.warning(f"Failed to fetch detailed content for page {page_id}: {str(e)}")
            return None


    def fetch_documents(self,
                       updated_since: Optional[datetime] = None,
                       updated_until: Optional[datetime] = None,
                       book_names: Optional[List[str]] = None,
                       chapter_id: Optional[str] = None,
                       progress_callback=None) -> Iterator[List[BookStackDocument]]:
        """
        Fetch documents from BookStack

        Args:
            updated_since: Only fetch documents updated after this date
            updated_until: Only fetch documents updated before this date
            book_names: List of book names to fetch documents from
            chapter_id: List of chapter urls to fetch documents from
            progress_callback: Progress callback function

        Yields:
            Batches of BookStackDocument objects
        """

        total_fetched = 0
        # Define content types to fetch
        content_types = []
        if self.include_shelves:
            content_types.append(("shelves", self._shelf_to_document, self.client.get_shelves))

        if self.include_book_to_chapters:
            # Special handling for chapters_of_books since it needs book_names parameter
            get_chapters_from_books_fetcher = partial(self.client.get_chapters_from_books, book_names)
            content_types.append(("chapters_of_books", self._chapter_to_document, get_chapters_from_books_fetcher))    
        if self.include_books:
            content_types.append(("books", self._book_to_document, self.client.get_books))
        if self.include_chapters:
            content_types.append(("chapters", self._chapter_to_document, self.client.get_chapters))

        if self.include_chapter_to_pages:
            print("chapter_id", chapter_id)
            get_chapter_to_pages_fetcher = partial(self.client.get_pages, chapter_id)
            content_types.append(("chapter_to_pages", self._page_to_document, get_chapter_to_pages_fetcher))    
        if self.include_pages:
            get_pages_fetcher = partial(self.client.get_pages, chapter_id)
            content_types.append(("pages", self._page_to_document, get_pages_fetcher))
            
        for content_type, converter, fetcher in content_types:
            if progress_callback:
                progress_callback(0, f"Fetching {content_type} from BookStack...")

            offset = 0
            while True:
                try:
                    items = fetcher(
                        count=self.batch_size,
                        offset=offset,
                        updated_since=updated_since,
                        updated_until=updated_until
                    )

                    if not items:
                        break
                    print("items", items)
                    # Convert to documents
                    documents = []
                    for item in items:
                        try:
                            doc = converter(item)
                            documents.append(doc)
                        except Exception as e:
                            logging.warning(f"Failed to convert {content_type} item {item.get('id')}: {str(e)}")
                            continue

                    if documents:
                        total_fetched += len(documents)
                        if progress_callback:
                            progress_callback(0, f"Fetched {total_fetched} {content_type} so far...")
                        yield documents

                    # Check if we got fewer items than requested (end of data)
                    if len(items) < self.batch_size:
                        break

                    offset += len(items)

                    # Rate limiting
                    time.sleep(0.2)

                except BookStackClientError as e:
                    logging.error(f"Error fetching {content_type}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
                except Exception as e:
                    logging.error(f"Unexpected error fetching {content_type}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break

        if progress_callback:
            progress_callback(1.0, f"Fetched {total_fetched} documents from BookStack")

    def create_virtual_file_binary(self, documents: List[BookStackDocument]) -> BytesIO:
        """
        Create a virtual file containing document data for RAGFlow processing

        Args:
            documents: List of BookStack documents

        Returns:
            BytesIO containing JSON data
        """
        data = {
            "source": "bookstack",
            "documents": []
        }

        for doc in documents:
            doc_data = {
                "id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "url": doc.url,
                "type": doc.doc_type,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "metadata": doc.metadata
            }
            data["documents"].append(doc_data)

        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        return BytesIO(json_data.encode('utf-8'))


if __name__ == "__main__":
    # Initialize connector   
    connector = BookStackConnector(
        base_url="https://knowledge.demo.securityzone.vn",
        token_id="GyEiV5NkEZ8ytCgSH4hXvB7BVwZQ9knP",
        token_secret="puKhXEpOAarRvE3A7jvDdUE9pB204fXt",
        batch_size=50,
        include_book_to_chapters=True,
    )

    print("global_config", "global_config")
    try:
        print("About to call fetch_documents()")
        result = connector.fetch_documents(book_names=["Thẻ tín dụng"])
        print("fetch_documents() returned:", type(result))
        # Since it returns an Iterator, we need to iterate
        for batch in result:
            print(f"Got batch with {len(batch)} documents")
            for doc in batch:
                print(doc.content)
            break  # Just test first batch
    except Exception as e:
        print(f"Error calling fetch_documents(): {e}")
        import traceback
        traceback.print_exc()