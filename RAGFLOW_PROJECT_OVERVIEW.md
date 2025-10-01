# RAGFlow Project Overview

## Executive Summary

RAGFlow is an open-source Retrieval-Augmented Generation (RAG) engine that combines deep document understanding with LLM capabilities to create a production-ready AI system. It offers a streamlined RAG workflow adaptable to enterprises of any scale, powered by a converged context engine and pre-built agent templates.

**Version:** 0.20.5
**License:** Apache 2.0
**Project URL:** https://ragflow.io/

## Project Architecture

### Core Components

```
RAGFlow
├── API Layer (Flask-based REST API)
├── Document Processing Pipeline (DeepDoc)
├── RAG Engine (Retrieval & Generation)
├── Storage Layer (MinIO, Elasticsearch/Infinity/OpenSearch)
├── Database Layer (MySQL/PostgreSQL)
├── Cache Layer (Redis/Valkey)
├── Web Frontend (React + UmiJS)
└── Task Processing System (Async Workers)
```

## Directory Structure

```
ragflow/
├── api/                    # Backend API server
│   ├── apps/              # API endpoints/controllers
│   ├── db/                # Database models and services
│   ├── utils/             # API utilities
│   └── ragflow_server.py  # Main server entry point
│
├── rag/                   # Core RAG engine
│   ├── app/               # Document parsers (naive, book, paper, etc.)
│   ├── flow/              # Pipeline components
│   ├── llm/               # LLM integrations
│   ├── nlp/               # NLP utilities
│   ├── connectors/        # External service connectors
│   └── svr/               # Task executor service
│
├── deepdoc/               # Document understanding engine
│   ├── parser/            # Format-specific parsers
│   └── vision/            # OCR and layout recognition
│
├── web/                   # Frontend application
│   ├── src/               # React source code
│   ├── pages/             # Page components
│   └── components/        # Reusable UI components
│
├── docker/                # Docker configurations
│   ├── docker-compose.yml
│   └── service_conf.yaml.template
│
├── agent/                 # Agent framework
├── graphrag/              # Graph-based RAG
├── plugin/                # Plugin system
└── sandbox/               # Code execution sandbox
```

## Main Entry Points

### 1. Backend Server
- **File:** `api/ragflow_server.py`
- **Purpose:** Flask-based HTTP server hosting REST APIs
- **Key Features:**
  - Handles authentication and session management
  - Routes API requests to appropriate controllers
  - Manages WebSocket connections for real-time updates
  - Initializes database and runtime configurations

### 2. Task Executor
- **File:** `rag/svr/task_executor.py`
- **Purpose:** Async worker for processing document tasks
- **Key Features:**
  - Processes document parsing tasks from Redis queue
  - Handles BookStack synchronization (line 151 reference)
  - Manages concurrent task execution with limiters
  - Supports various parser types (naive, book, paper, etc.)

## API Structure

### Key API Endpoints (from `api/apps/`)

1. **Knowledge Base Management** (`kb_app.py`)
   - `/create` - Create new knowledge base
   - `/update` - Update KB configuration
   - `/sync_bookstack` - Sync BookStack content (line 177)
   - `/list` - List all knowledge bases
   - `/rm` - Delete knowledge base

2. **Document Management** (`document_app.py`)
   - `/upload` - Upload documents
   - `/parse` - Parse document content
   - `/chunk` - Create document chunks
   - `/list` - List documents in KB

3. **Chat/Conversation** (`conversation_app.py`)
   - `/create` - Create conversation
   - `/answer` - Get RAG-based answers
   - `/stream` - Stream responses

4. **LLM Configuration** (`llm_app.py`)
   - `/list` - List available models
   - `/test` - Test LLM connectivity

## Core RAG Functionality

### Document Processing Pipeline

1. **Input Processing**
   - Supports multiple formats: PDF, DOCX, PPTX, Excel, images, HTML, Markdown
   - Special parsers for specific content types (papers, books, resumes)

2. **Chunking Strategies**
   - Template-based chunking
   - Intelligent document segmentation
   - Preserves document structure and context

3. **Embedding & Indexing**
   - Multiple embedding model support
   - Vector storage in Elasticsearch/Infinity/OpenSearch
   - Hybrid search (keyword + semantic)

4. **Retrieval**
   - Multi-modal retrieval
   - Reranking capabilities
   - Context fusion from multiple sources

5. **Generation**
   - LLM integration (OpenAI, Anthropic, local models)
   - Grounded citations
   - Hallucination reduction

## Database Models

### Key Entities
- **Knowledgebase** - Container for documents
- **Document** - Individual files/content
- **Task** - Async processing tasks
- **User/Tenant** - Multi-tenancy support
- **Dialog/Conversation** - Chat sessions

## External Integrations

### BookStack Integration (Line 151 Reference)
- **Location:** `api/apps/kb_app.py:151`
- **Task Type:** `bookstack_chapter_doc`
- **Connector:** `rag/connectors/bookstack_connector.py`
- **Functionality:**
  - Fetches content from BookStack instances
  - Converts BookStack pages/chapters to RAG chunks
  - Supports batch synchronization
  - Maintains content freshness

### Supported Storage Backends
- **MinIO** - Object storage
- **Elasticsearch** - Default vector store
- **Infinity** - Alternative vector database
- **OpenSearch** - Elasticsearch alternative
- **Azure Blob Storage**
- **AWS S3**
- **Aliyun OSS**

## Key Technologies

### Backend Stack
- **Python 3.10-3.12**
- **Flask** - Web framework
- **Peewee** - ORM
- **Redis/Valkey** - Queue and cache
- **Trio** - Async concurrency
- **NumPy/Pandas** - Data processing

### Frontend Stack
- **React 18**
- **UmiJS** - React framework
- **Ant Design** - UI components
- **TypeScript**

### AI/ML Libraries
- **Transformers** - NLP models
- **ONNX Runtime** - Model inference
- **OpenCV** - Image processing
- **Tiktoken** - Tokenization
- **LangFuse** - Observability

## Configuration

### Main Configuration Files
1. **docker/.env** - Environment variables
2. **docker/service_conf.yaml.template** - Service configurations
3. **conf/llm_factories.json** - LLM provider configurations
4. **pyproject.toml** - Python dependencies

### Key Environment Variables
- `DB_TYPE` - Database type (mysql/postgresql)
- `DOC_ENGINE` - Document store (elasticsearch/infinity/opensearch)
- `RAGFLOW_IMAGE` - Docker image version
- `MAX_CONCURRENT_TASKS` - Task parallelism
- `SANDBOX_ENABLED` - Code execution sandbox

## Deployment Architecture

### Docker Services
1. **ragflow-server** - Main application server
2. **ragflow-mysql** - Database
3. **ragflow-es-01** - Elasticsearch (optional)
4. **ragflow-infinity** - Infinity vector DB (optional)
5. **ragflow-minio** - Object storage
6. **ragflow-redis** - Cache and queue
7. **ragflow-sandbox-executor** - Code execution (optional)

### Ports
- **80/443** - Web interface
- **9380** - API server
- **9382** - MCP server (optional)
- **3306** - MySQL
- **9200** - Elasticsearch
- **9000** - MinIO

## Recent Updates (2025)

- Support for latest LLMs (GPT-5, Kimi K2, Grok 4)
- Agentic workflow and MCP integration
- Python/JavaScript code executor
- Cross-language query support
- Multi-modal document understanding
- Deep Research reasoning capability
- Enhanced document layout analysis

## Security Features

- Token-based authentication
- API key management
- Multi-tenancy isolation
- Sandboxed code execution
- SSL/TLS support
- Role-based access control

## Performance Optimizations

- Concurrent task processing
- Batch embedding generation
- Connection pooling
- Redis caching
- Lazy loading
- Stream processing for large documents

## Monitoring & Debugging

- Comprehensive logging system
- Memory profiling (tracemalloc)
- Task progress tracking
- Health check endpoints
- Debug mode support
- Performance metrics

## Development Workflow

### Setup
1. Clone repository
2. Install dependencies via `uv`
3. Configure environment variables
4. Start Docker services
5. Run backend server
6. Start frontend dev server

### Testing
- Unit tests with pytest
- Integration tests
- Performance benchmarks
- End-to-end testing

## Summary

RAGFlow is a comprehensive, production-ready RAG system that excels in:
- **Deep document understanding** with specialized parsers
- **Flexible deployment** with multiple storage options
- **Enterprise features** like multi-tenancy and security
- **Extensibility** through plugins and connectors
- **Modern architecture** with microservices and async processing

The BookStack integration at line 151 demonstrates the system's ability to connect with external knowledge management systems, creating tasks for fetching and processing content from BookStack instances into the RAG pipeline.