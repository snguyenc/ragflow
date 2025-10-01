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
import time
from typing import Any, Dict, List, Optional
import requests
from datetime import datetime


class BookStackClientError(Exception):
    """Base exception for BookStack client errors"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class BookStackClient:
    """
    BookStack API Client for RAGFlow

    Provides interface to connect and fetch data from BookStack knowledge management system.
    Supports authentication via API tokens and fetching of books, chapters, shelves, and pages.
    """

    def __init__(self, base_url: str, token_id: str, token_secret: str, timeout: int = 30):
        """
        Initialize BookStack client

        Args:
            base_url: BookStack instance URL
            token_id: API token ID
            token_secret: API token secret
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.token_id = token_id
        self.token_secret = token_secret
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self._get_headers())

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            'Authorization': f'Token {self.token_id}:{self.token_secret}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def _build_api_url(self, endpoint: str) -> str:
        """Build full API URL"""
        return f"{self.base_url}/api/{endpoint.lstrip('/')}"

    def build_app_url(self, path: str) -> str:
        """Build app URL for web access"""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _make_request(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            BookStackClientError: On API errors
        """
        url = self._build_api_url(endpoint)

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)

            # Parse JSON response
            try:
                data = response.json()
            except ValueError:
                data = {}

            # Check for errors
            if response.status_code >= 400:
                error_msg = data.get('error', {}).get('message', response.reason)
                logging.error(f"BookStack API error: {response.status_code} - {error_msg}")
                raise BookStackClientError(
                    f"BookStack API error: {error_msg}",
                    status_code=response.status_code
                )

            return data

        except requests.RequestException as e:
            logging.error(f"BookStack request failed: {str(e)}")
            raise BookStackClientError(f"Request failed: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test connection to BookStack API

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._make_request('/books', params={'count': '1'})
            return True
        except BookStackClientError:
            return False

    def get_books(self, count: int = 50, offset: int = 0,
                  updated_since: Optional[datetime] = None,
                  updated_until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch books from BookStack

        Args:
            count: Number of books to fetch
            offset: Offset for pagination
            updated_since: Only fetch books updated after this date
            updated_until: Only fetch books updated before this date

        Returns:
            List of book data
        """
        params = {
            'count': str(count),
            'offset': str(offset),
            'sort': '+id'
        }

        if updated_since:
            params['filter[updated_at:gte]'] = updated_since.strftime('%Y-%m-%d')
        if updated_until:
            params['filter[updated_at:lte]'] = updated_until.strftime('%Y-%m-%d')

        response = self._make_request('/books', params)
        return response.get('data', [])

    def get_chapters_from_books(self, book_names: List[str], count: int = 50, offset: int = 0,
                  updated_since: Optional[datetime] = None,
                  updated_until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch chapters from BookStack

        Args:
            count: Number of books to fetch
            offset: Offset for pagination
            updated_since: Only fetch books updated after this date
            updated_until: Only fetch books updated before this date

        Returns:
            List of book data
        """

        book_ids = []
        books = self.get_books(count=count, offset=offset, updated_since=updated_since, updated_until=updated_until)
        book_map = {book['name']: book['id'] for book in books}

        for book_name in book_names:
            if book_name in book_map:
                book_ids.append(book_map[book_name])
            else:
                logging.warning(f"Book '{book_name}' not found")

        if not book_ids:
            return []
        
        params = {
            'count': str(count),
            'offset': str(offset),
            'sort': '+id'
        }

        if updated_since:
            params['filter[updated_at:gte]'] = updated_since.strftime('%Y-%m-%d')
        if updated_until:
            params['filter[updated_at:lte]'] = updated_until.strftime('%Y-%m-%d')

        chapters = self.get_chapters(count=count, offset=offset, updated_since=updated_since, updated_until=updated_until)
        chapters = [chapter for chapter in chapters if chapter['book_id'] in book_ids]   

        return chapters

    def get_chapters(self, count: int = 50, offset: int = 0,
                    updated_since: Optional[datetime] = None,
                    updated_until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch chapters from BookStack

        Args:
            count: Number of chapters to fetch
            offset: Offset for pagination
            updated_since: Only fetch chapters updated after this date
            updated_until: Only fetch chapters updated before this date

        Returns:
            List of chapter data
        """
        params = {
            'count': str(count),
            'offset': str(offset),
            'sort': '+id'
        }

        if updated_since:
            params['filter[updated_at:gte]'] = updated_since.strftime('%Y-%m-%d')
        if updated_until:
            params['filter[updated_at:lte]'] = updated_until.strftime('%Y-%m-%d')

        response = self._make_request('/chapters', params)
        return response.get('data', [])

    def get_shelves(self, count: int = 50, offset: int = 0,
                   updated_since: Optional[datetime] = None,
                   updated_until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch shelves from BookStack

        Args:
            count: Number of shelves to fetch
            offset: Offset for pagination
            updated_since: Only fetch shelves updated after this date
            updated_until: Only fetch shelves updated before this date

        Returns:
            List of shelf data
        """
        params = {
            'count': str(count),
            'offset': str(offset),
            'sort': '+id'
        }

        if updated_since:
            params['filter[updated_at:gte]'] = updated_since.strftime('%Y-%m-%d')
        if updated_until:
            params['filter[updated_at:lte]'] = updated_until.strftime('%Y-%m-%d')

        response = self._make_request('/shelves', params)
        return response.get('data', [])

    def get_pages(self, chapter_id: Optional[str] = None, count: int = 50, offset: int = 0,
                 updated_since: Optional[datetime] = None,
                 updated_until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch pages from BookStack

        Args:
            count: Number of pages to fetch
            offset: Offset for pagination
            updated_since: Only fetch pages updated after this date
            updated_until: Only fetch pages updated before this date

        Returns:
            List of page data
        """
        params = {
            'count': str(count),
            'offset': str(offset),
            'sort': '+id'
        }

        if updated_since:
            params['filter[updated_at:gte]'] = updated_since.strftime('%Y-%m-%d')
        if updated_until:
            params['filter[updated_at:lte]'] = updated_until.strftime('%Y-%m-%d')
        if chapter_id:
            params['filter[chapter_id]'] = chapter_id

        response = self._make_request('/pages', params)
        return response.get('data', [])


    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Fetch detailed page content

        Args:
            page_id: Page ID to fetch

        Returns:
            Page content data including HTML
        """
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        return self._make_request(f'/pages/{page_id}')

    def get_book_content(self, book_id: str) -> Dict[str, Any]:
        """
        Fetch detailed book content

        Args:
            book_id: Book ID to fetch

        Returns:
            Book content data
        """
        return self._make_request(f'/books/{book_id}')

    def get_chapter_content(self, chapter_id: str) -> Dict[str, Any]:
        """
        Fetch detailed chapter content

        Args:
            chapter_id: Chapter ID to fetch

        Returns:
            Chapter content data
        """
        return self._make_request(f'/chapters/{chapter_id}')

    def get_shelf_content(self, shelf_id: str) -> Dict[str, Any]:
        """
        Fetch detailed shelf content

        Args:
            shelf_id: Shelf ID to fetch

        Returns:
            Shelf content data
        """
        return self._make_request(f'/shelves/{shelf_id}')