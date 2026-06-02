# Memory System Architecture Research & Pipeline

## Project Overview
A comprehensive memory system with graph-based storage, vector embeddings, and style learning capabilities.

## Core System Analysis

### 1. Memory System Core (`memory_system.py`)

**Purpose**: Central orchestrator that coordinates all memory operations including storage, retrieval, and learning.

**Key Components**:
- **MemorySystem Class**: Main interface for memory operations
- **Initialization**: Sets up graph store, vector store, entity index, and style components
- **Memory Operations**: Add, search, update, delete memories with automatic learning
- **Style Learning Integration**: Automatic style analysis and profile updates

**Core Methods**:
- `add(messages, user_id, metadata)`: Stores new memories with automatic deduplication and style learning
- `search(query, user_id, limit)`: Retrieves relevant memories using hybrid search (vector + graph)
- `get_all(user_id)`: Fetches all memories for a user
- `update(memory_id, data)`: Updates existing memories
- `delete(memory_id)`: Removes memories and updates indexes
- `delete_all(user_id)`: Clears all user memories
- `reset()`: Complete system reset

**Architecture Features**:
- **Hybrid Storage**: Combines graph-based relationships with vector similarity
- **Automatic Learning**: Style analysis and pattern recognition on every memory addition
- **Multi-Index Support**: Entity index, vector store, and graph store working together
- **Metadata Management**: Rich metadata storage for context and relationships
- **Error Handling**: Comprehensive error management and logging

**Integration Points**:
- Graph Store for relationship mapping
- Vector Store for semantic similarity
- Entity Index for entity tracking
- Style System for communication pattern learning
- Local LLM for processing and analysis

**Data Flow**:
1. Input → Memory Processing → Deduplication Check
2. Vector Embedding Generation → Storage in Vector Store
3. Entity Extraction → Entity Index Update
4. Relationship Analysis → Graph Store Update
5. Style Analysis → Profile Learning Update
6. Metadata Storage → Complete Memory Record

---

### 2. Offline Memory Engine (`memory.py`)

**Purpose**: Core implementation of the offline memory system with rule-based entity extraction and graph reasoning.

**Key Architecture Features**:
- **Hybrid Retrieval System**: Combines FAISS vector search with graph k-hop expansion
- **Rule-Based Entity Extraction**: Deterministic parsing without LLM prompts for speed
- **Dynamic Relationship Extraction**: LLM-powered relationship detection with graph context
- **Similarity-Based Merging**: Automatic entity and relationship deduplication
- **Multi-Index Coordination**: Synchronizes vector store, graph store, and entity index

**Core Components**:

#### OfflineMemory Class
- **Initialization**: Sets up all storage backends and processing components
- **Caching System**: Neighbor and degree caches for traversal optimization
- **Multi-Store Architecture**: Coordinates embedder, vectors, graph, entities, LLM, and node vectors

#### Memory Storage Pipeline (`add` method):
1. **Input Processing**: Handles both string and message list inputs
2. **Similarity Checking**: Searches for existing similar memories using vector similarity
3. **Merge/Update Logic**: Updates existing memories if similarity threshold met
4. **Entity & Graph Indexing**: Extracts entities and relationships with LLM assistance
5. **Vector Storage**: Stores embeddings in FAISS with rich metadata
6. **History Tracking**: Records all operations in SQLite for auditability

#### Advanced Relationship Extraction:
- **Context-Aware Processing**: Builds graph context before LLM relationship extraction
- **Multi-Triplet Extraction**: Extracts multiple (subject, relation, object) triplets per input
- **Deduplication**: Prevents duplicate relationships using normalized keys
- **Temporal Attachment**: Detects time references and attaches temporal properties
- **Entity Resolution**: Maps possessive patterns to actual entity names

#### Similarity-Based Entity Merging:
- **Dynamic Entity Matching**: Finds similar entities using configurable thresholds
- **Relationship Consolidation**: Merges similar relationship types
- **FAISS Update Propagation**: Updates vector embeddings when entities merge
- **History Preservation**: Maintains audit trail of merge operations

#### Hybrid Search System (`search` method):
1. **Vector Search**: Initial FAISS search with lexical features
2. **Entity Detection**: Extracts entities from query using pattern matching
3. **Graph Expansion**: K-hop traversal around detected entities
4. **Memory Collection**: Gathers related memories via entity index
5. **Traversal Enhancement**: Advanced graph traversal for comprehensive results
6. **Hybrid Ranking**: Combines vector scores with graph relevance

#### Graph Traversal (`traverse` method):
- **Multi-Hop Exploration**: Configurable depth graph traversal
- **Scoring System**: Distance-based scoring with degree penalties
- **Path Context**: Embeds relationship paths for semantic relevance
- **Entity Promotion**: Promotes important nodes based on connectivity

**Technical Optimizations**:
- **Batch Processing**: Batch embedding for performance
- **Cache Management**: Neighbor and degree caching for speed
- **Memory Efficient**: Limits candidate pools and uses efficient data structures
- **Error Resilience**: Comprehensive error handling with graceful degradation

**Integration Interfaces**:
- **Vector Store**: FAISS-based semantic search
- **Graph Store**: Neo4j-style relationship storage
- **Entity Index**: Fast entity-to-memory mapping
- **Node Vectors**: Embedded entity and path representations
- **SQLite History**: Complete operation audit trail
- **Local LLM**: Relationship extraction and processing

---

### 3. Graph Storage Engine (`graph_store.py`)

**Purpose**: SQLite-backed property graph store providing relationship storage and graph traversal capabilities.

**Architecture Design**:
- **Property Graph Model**: Supports nodes with properties and labeled, weighted edges
- **SQLite Backend**: High-performance local storage with full ACID transactions
- **Thread-Safe Operations**: Comprehensive locking for concurrent access
- **Flexible Schema**: Dynamic node/edge properties with extensible metadata

**Core Database Schema**:

#### Tables Structure:
1. **nodes**: `id, name, label, props, cluster_id, created_at, updated_at`
2. **edges**: `source, relationship, destination, memory_id, source_id, destination_id, created_at, updated_at, weight`
3. **node_labels**: Multi-label support for nodes
4. **node_props**: Key-value properties for nodes
5. **edge_props**: Key-value properties for edges

#### Index Strategy:
- **Node Indexes**: name, label, unique(name, label)
- **Edge Indexes**: source, destination, relationship, memory_id, source_id, destination_id
- **Property Indexes**: Efficient key-based property lookups
- **Unique Constraints**: Prevents duplicate relationships

**Core Operations**:

#### Node Management:
- **upsert_node()**: Create or update nodes with properties
- **get_node_by_name()**: Retrieve nodes by name
- **node_has_label()**: Check node label membership
- **set_node_prop()/get_node_prop()**: Property management
- **delete_node()**: Node removal with edge cleanup

#### Edge Operations:
- **add()/upsert()**: Relationship creation with weight accumulation
- **update()**: Modify relationship properties and weights
- **delete()**: Remove specific relationships
- **set_edge_prop()/get_edge_prop()**: Edge property management
- **query_edges()**: Flexible edge querying with filters

#### Graph Traversal:
- **neighbors()**: Get adjacent nodes with direction control (in/out/both)
- **k_hop()**: Multi-hop traversal with configurable depth and limits
- **find()**: Exact relationship lookup
- **edges_for_node()**: Get all edges for a specific node

#### Advanced Features:

**Clustering System**:
- **recompute_clusters()**: Connected component analysis using Union-Find
- **cluster_id**: Automatic cluster assignment for graph components
- **Component Tracking**: Maintains graph connectivity information

**Entity & Relationship Similarity**:
- **find_similar_entities()**: Multi-method similarity detection
  - Exact case-insensitive matching
  - Substring containment analysis  
  - Sequence similarity (typo tolerance)
  - Word-level similarity for multi-word entities
- **find_similar_relationships()**: Relationship type similarity
- **Configurable Thresholds**: Adjustable similarity scoring

**Merge Operations**:
- **merge_entities()**: Consolidate duplicate entities
  - Updates all edge references
  - Handles node_id reassignment
  - Removes duplicate relationships
  - Preserves relationship history
- **merge_relationships()**: Combine similar relationship types
- **Transaction Safety**: Full rollback on merge failures

**Memory Integration**:
- **memory_id Tracking**: Links edges to specific memory records
- **delete_by_memory_id()**: Cascading memory deletion
- **Temporal Properties**: Automatic time window detection and attachment
- **Audit Trail**: Complete operation history

**Performance Optimizations**:
- **Batch Operations**: Efficient bulk inserts and updates
- **Connection Pooling**: Thread-safe SQLite connection management
- **Index Optimization**: Strategic indexing for common query patterns
- **Lazy Loading**: On-demand relationship expansion

**Transaction Management**:
- **Context Manager**: Automatic transaction handling
- **ACID Compliance**: Full transactional integrity
- **Error Recovery**: Automatic rollback on failures
- **Deadlock Prevention**: Consistent locking order

---

### 4. Vector Storage Engine (`vector_store.py`)

**Purpose**: FAISS-based vector storage system providing semantic similarity search and efficient vector operations.

**Architecture Design**:
- **FAISS Backend**: High-performance approximate nearest neighbor search
- **Flexible Index Types**: Support for FLAT, HNSW, and IVF indexes
- **Metadata Management**: Rich payload storage with filtering capabilities
- **Distance Strategies**: Inner product, cosine similarity, and L2 distance

**Core Components**:

#### Index Configuration:
- **FLAT Index**: Exact search for smaller datasets
- **HNSW Index**: Hierarchical navigable small world for fast approximate search
- **IVF Index**: Inverted file index for large-scale datasets
- **Dynamic Training**: Automatic index training for IVF-based indexes

#### Vector Operations:

**Storage Management**:
- **add()**: Batch vector insertion with automatic normalization
- **update()**: In-place vector and payload updates
- **delete()**: Vector removal with mapping cleanup
- **get()**: Direct vector record retrieval
- **list()**: Paginated vector listing

**Search Capabilities**:
- **search()**: K-nearest neighbor search with metadata filtering
- **Hybrid Filtering**: Top-level and nested metadata filtering
- **Dynamic K**: Intelligent candidate pool sizing for filtered results
- **Score Normalization**: Consistent scoring across distance strategies

#### Data Management:

**Persistence Layer**:
- **Dual Storage**: Separate FAISS index and pickle-based metadata
- **Automatic Saving**: Persistent storage after each operation
- **Error Recovery**: Graceful handling of corrupted indexes
- **Version Management**: Backward-compatible loading

**Memory Mapping**:
- **id_to_idx**: Vector ID to FAISS index mapping
- **idx_to_id**: FAISS index to vector ID reverse mapping
- **docstore**: Rich metadata and payload storage
- **Consistency Guarantees**: Synchronized mapping updates

#### Performance Optimizations:

**Normalization Strategy**:
- **Cosine Similarity**: Automatic L2 normalization for cosine distance
- **Inner Product**: Direct vector operations for maximum speed
- **Configurable**: Distance strategy selection per use case

**Index Optimization**:
- **HNSW Tuning**: Optimized efSearch parameters for speed/accuracy balance
- **IVF Configuration**: Adaptive nprobe settings for recall optimization
- **Memory Efficiency**: Optimized data structures and batch operations

**Maintenance Operations**:
- **reset_index()**: Complete index reconstruction
- **rebuild_from_docstore()**: Recovery from metadata after index corruption
- **Garbage Collection**: Efficient handling of deleted vectors

#### Integration Features:

**Filtering System**:
- **Nested Metadata**: Support for complex metadata hierarchies
- **List Filters**: Membership testing for categorical data
- **Exact Matching**: Precise metadata filtering
- **Performance Buffers**: Oversized candidate pools for effective filtering

**Vector Record Model**:
- **VectorRecord Class**: Encapsulation of ID and payload
- **Rich Payloads**: Support for arbitrary metadata structures
- **Type Safety**: Consistent data type handling

**Error Handling**:
- **Graceful Degradation**: Continued operation with corrupted data
- **Logging Integration**: Comprehensive operation logging
- **Recovery Mechanisms**: Automatic index rebuilding on failures

---

### 5. Entity Index System (`entity_index.py`)

**Purpose**: Lightweight entity recognition and memory linking system for fast entity-to-memory mapping.

**Architecture Design**:
- **SQLite Backend**: High-performance local entity storage
- **Pattern-Based Extraction**: Deterministic entity recognition without LLM overhead
- **Bidirectional Mapping**: Efficient entity-to-memory and memory-to-entity lookups
- **Alias Resolution**: Canonical entity name mapping and normalization

**Core Components**:

#### Database Schema:
- **entities**: `entity TEXT PRIMARY KEY` - Master entity registry
- **aliases**: `alias TEXT, canonical TEXT` - Entity name normalization
- **memory_entities**: `memory_id TEXT, entity TEXT` - Bidirectional linking
- **Indexes**: Optimized for memory_id and entity lookups

#### Entity Extraction Engine:

**Pattern Recognition**:
- **ENTITY_PATTERN**: Regex-based proper noun detection (`[A-Z][a-zA-Z0-9_\-]+`)
- **Quoted Phrase Extraction**: Captures explicitly quoted entity names
- **Capitalization Normalization**: Consistent entity name formatting
- **Deduplication**: Automatic removal of duplicate entities per memory

**Recognition Rules**:
- **Proper Case Detection**: Identifies capitalized words as potential entities
- **Multi-word Entities**: Extracts first token of quoted phrases
- **Additional Entity Support**: Manual entity injection for special cases
- **Heuristic Filtering**: Removes common false positives

#### Core Operations:

**Memory-Entity Linking**:
- **add_memory_entities()**: Links extracted entities to memory records
- **entities_for_memory()**: Gets all entities mentioned in a specific memory
- **memories_for_entity()**: Finds all memories mentioning a specific entity
- **Batch Processing**: Efficient bulk entity-memory associations

**Entity Management**:
- **Automatic Registration**: Entities auto-added to master registry
- **Conflict Prevention**: INSERT OR IGNORE for duplicate entity prevention
- **Thread Safety**: Comprehensive locking for concurrent access
- **Transaction Integrity**: Atomic operations for consistency

#### Alias Resolution System:

**Name Normalization**:
- **add_alias()**: Create alias-to-canonical mappings
- **resolve()**: Convert alias names to canonical forms
- **Case Handling**: Intelligent case variation management
- **Replacement Strategy**: OR REPLACE for alias updates

**Use Cases**:
- **Nickname Resolution**: "Bob" → "Robert"
- **Variation Handling**: "AI" → "Artificial Intelligence"
- **Typo Correction**: Common misspelling normalization
- **Format Standardization**: Consistent entity name formats

#### Performance Characteristics:

**Lookup Optimization**:
- **Indexed Queries**: Fast entity and memory_id lookups
- **Minimal Processing**: Regex-only entity extraction
- **Memory Efficient**: Lightweight SQLite storage
- **Batch Operations**: Efficient bulk entity processing

**Scalability Features**:
- **Thread Safety**: Concurrent operation support
- **Connection Pooling**: Efficient database connection management
- **Index Strategy**: Optimized for common query patterns
- **Minimal Overhead**: Fast entity extraction without LLM calls

#### Integration Points:

**Memory System Integration**:
- **Automatic Extraction**: Called during memory storage operations
- **Bidirectional Queries**: Supports both entity→memory and memory→entity lookups
- **Graph Integration**: Provides entity lists for graph relationship building
- **Search Enhancement**: Enables entity-based memory filtering

**Error Handling**:
- **Graceful Failures**: Continues operation if entity extraction fails
- **Data Validation**: Ensures entity name consistency
- **Recovery Support**: Rebuilds entity-memory links on corruption

---

### 6. Local LLM Engine (`local_llm.py`)

**Purpose**: Multi-backend LLM inference system optimized for local deployment with Intel and GGUF support.

**Architecture Design**:
- **Multi-Backend Support**: Intel ipex-llm and llama-cpp-python backends
- **Automatic Backend Selection**: Intelligent model format detection and optimization
- **Resource-Aware Inference**: Adaptive configuration based on system resources
- **Streaming Support**: Real-time token generation for interactive applications

**Core Components**:

#### Backend Selection System:

**Automatic Detection**:
- **GGUF Detection**: Scans for .gguf files in model directory
- **Intel Optimization**: ipex-llm INT4 quantization for CPU/GPU acceleration
- **Fallback Strategy**: Graceful degradation to available backends
- **Configuration Override**: Manual backend selection support

**Backend Implementations**:
- **GGUF Backend**: llama-cpp-python for GGUF quantized models
- **Intel Backend**: ipex-llm AutoModelForCausalLM with 4-bit optimization
- **Auto Mode**: Intelligent selection based on model format and system

#### Resource Management:

**Adaptive Configuration**:
- **Memory Detection**: psutil-based system memory analysis
- **CPU Optimization**: Thread count based on available CPU cores
- **Context Window Scaling**: Dynamic context size based on available RAM
- **Batch Size Tuning**: Adaptive batch processing for performance

**GGUF Resource Mapping**:
- **Low Memory (<4GB)**: ctx=2048, batch=128, threads=4
- **Medium Memory (4-8GB)**: ctx=3072, batch=256, threads=6
- **High Memory (>8GB)**: ctx=4096, batch=512, threads=8
- **GPU Acceleration**: Automatic layer offloading when CUDA available

**Intel Backend Optimization**:
- **4-bit Quantization**: INT4 model compression for faster inference
- **Model Optimization**: Built-in optimization pass for Intel hardware
- **Cache Management**: Efficient KV-cache for multi-turn conversations
- **Trust Remote Code**: Support for custom model architectures

#### Generation Interface:

**Text Generation**:
- **generate()**: Single-prompt text generation
- **generate_messages()**: Chat-based message generation
- **generate_messages_stream()**: Real-time streaming generation
- **Configurable Parameters**: Temperature, top_p, max_new_tokens

**Message Formatting**:
- **Llama 3.1 Instruct Format**: Standardized chat template
- **System Message Support**: Proper system prompt integration
- **Conversation Management**: Automatic conversation history truncation
- **Token Optimization**: Efficient prompt construction

#### Streaming Implementation:

**GGUF Streaming**:
- **Native Streaming**: llama-cpp-python streaming API
- **Token-by-Token**: Real-time token generation
- **Stop Token Management**: Automatic stopping on completion tokens
- **Error Recovery**: Fallback to non-streaming generation

**Intel Streaming**:
- **TextIteratorStreamer**: HuggingFace streaming integration
- **Threading Support**: Non-blocking generation with background threads
- **Character Fallback**: Character-by-character streaming when streamer unavailable
- **Memory Efficient**: Minimal memory overhead during streaming

#### Performance Features:

**Model Loading Optimization**:
- **Lazy Imports**: Runtime import loading to avoid dependency conflicts
- **Memory Mapping**: Efficient model file access with mmap
- **Fast Startup**: Minimal initialization overhead
- **Error Handling**: Graceful fallback on loading failures

**Inference Optimization**:
- **torch.inference_mode()**: Disabled gradient computation for speed
- **Conversation Truncation**: Last 10 pairs (20 messages) for context efficiency
- **Sequential Processing**: Fast message formatting
- **Device Resolution**: Automatic CUDA/CPU device selection

#### Integration Features:

**Template System**:
- **Format Consistency**: Unified message formatting across backends
- **Header Management**: Proper role-based message headers
- **Context Preservation**: System message priority handling
- **EOT Token Management**: Proper conversation termination

**Error Handling**:
- **Backend Fallbacks**: Automatic fallback between generation methods
- **Resource Limits**: Graceful handling of memory constraints
- **Import Errors**: Robust handling of missing dependencies
- **Runtime Errors**: Comprehensive error reporting and recovery

---

### 7. Local Embedding Engine (`local_embedding.py`)

**Purpose**: Advanced embedding system using BGE-M3 with multi-modal retrieval and graph-enhanced reranking.

**Architecture Design**:
- **BGE-M3 Integration**: State-of-the-art multi-modal embedding model
- **Hybrid Retrieval**: Dense, sparse, and ColBERT scoring modes
- **Graph-Enhanced Ranking**: Integration with knowledge graph context
- **Adaptive Fusion**: Query-aware scoring combination strategies

**Core Components**:

#### BGE-M3 Model Integration:

**Model Configuration**:
- **FlagEmbedding Backend**: BGE-M3FlagModel with FP16 optimization
- **Offline Mode**: TRANSFORMERS_OFFLINE environment for local inference
- **Model Path Resolution**: Automatic HuggingFace cache detection
- **Dimension Management**: Configurable embedding dimensions with padding/truncation

**Embedding Operations**:
- **embed()**: Single text embedding with normalization
- **embed_many()**: Batch embedding for efficiency
- **embed_with_lexical()**: Combined dense and sparse embedding extraction
- **Empty Text Handling**: Zero-vector fallback for empty inputs

#### Multi-Modal Retrieval System:

**Scoring Modes**:
- **Dense Vectors**: Semantic similarity via dense embeddings
- **Sparse Vectors**: BM25-style lexical matching
- **ColBERT**: Fine-grained token-level interaction scoring
- **Hybrid Fusion**: Weighted combination of all modes

**Lexical Processing**:
- **lexical_score()**: BM25-style similarity computation
- **lexical_scores()**: Batch lexical scoring
- **Token-Level Matching**: Fine-grained lexical overlap analysis

#### Advanced Reranking System:

**Query-Adaptive Fusion**:
- **Query Type Detection**: Factual vs conceptual query classification
- **Content Analysis**: Word overlap and exact match detection
- **Adaptive Weighting**: Dynamic score combination based on query characteristics
- **Multiple Fusion Strategies**: Linear, geometric, harmonic, and max fusion

**Fusion Strategy Selection**:
- **Factual Queries**: Boost sparse scoring for exact matches
- **Conceptual Queries**: Emphasize dense semantic understanding
- **Short Queries**: ColBERT for fine-grained matching
- **Long Queries**: Dense vectors for context understanding

**Advanced Scoring Techniques**:
- **Linear Fusion**: Weighted average of all modes
- **Geometric Fusion**: Balanced multiplicative combination
- **Harmonic Fusion**: Conservative scoring for high precision
- **Max Fusion**: Best-of-breed approach
- **Quality Boosting**: Confidence-based score enhancement

#### Graph-Enhanced Reranking:

**Context Integration**:
- **Entity Enhancement**: Memory enrichment with entity context
- **Relationship Context**: Graph relationship integration in text
- **Graph Confidence**: Knowledge graph reliability scoring
- **Traversal Strength**: Multi-hop path confidence measurement

**Graph-Aware Boosting**:
- **Entity Overlap Boost**: Bonus for entity matches (up to 0.3)
- **Confidence Boost**: Graph reliability enhancement (up to 0.2)
- **Traversal Boost**: Multi-hop path strength bonus (up to 0.25)
- **Mode-Specific Enhancement**: Targeted boosting per scoring mode

**Enhanced Text Representation**:
- **Entity Context**: `[ENTITIES: entity1, entity2, ...]` format
- **Relationship Context**: `[RELATIONSHIPS: src rel dst; ...]` format
- **Structured Enhancement**: BGE-M3 optimized context formatting
- **Fallback Handling**: Graceful degradation to standard reranking

#### Performance Optimizations:

**Batch Processing**:
- **Vectorized Operations**: NumPy-based efficient computation
- **Memory Management**: FP32 normalization and padding
- **Dimension Consistency**: Automatic vector size management
- **Error Recovery**: Graceful handling of embedding failures

**Model Optimization**:
- **FP16 Inference**: Half-precision for speed
- **Max Length Control**: Configurable token limits
- **Offline Operation**: No internet dependencies
- **Cache Utilization**: HuggingFace model caching

#### Integration Features:

**Scoring Interface**:
- **pair_scores_m3()**: Multi-modal pairwise scoring
- **advanced_rerank_m3()**: Sophisticated reranking with fusion
- **graph_enhanced_rerank_m3()**: Knowledge graph integrated reranking
- **Detailed Breakdowns**: Comprehensive scoring analysis

**Configuration Management**:
- **Flexible Weights**: Customizable fusion weights
- **Passage Length**: Configurable maximum passage length
- **Query Analysis**: Automatic query characteristic detection
- **Performance Tuning**: Optimized settings for different scenarios

**Error Handling**:
- **Graceful Fallbacks**: Automatic degradation on component failures
- **Comprehensive Logging**: Detailed error reporting
- **Recovery Mechanisms**: Fallback to simpler scoring methods
- **Validation**: Input validation and sanitization

---

### 8. Node Vector Index (`node_vectors.py`)

**Purpose**: Specialized vector index for graph entities and relationship paths with semantic search capabilities.

**Architecture Design**:
- **LocalVectorStore Wrapper**: Specialized configuration for node/path vectors
- **Entity Embeddings**: Vector representations of graph entities
- **Path Embeddings**: Relationship path vector storage
- **Semantic Node Search**: Vector-based entity and path retrieval

**Core Components**:

#### Configuration Management:
- **Dedicated Collection**: Separate namespace for node vectors
- **Normalized Vectors**: Automatic L2 normalization for consistency
- **Flexible Index Types**: FLAT, HNSW, IVF support via LocalVectorStore
- **Configurable Dimensions**: Adaptive to embedding model dimensions

#### Vector Operations:
- **upsert()**: Atomic delete-then-add operations for consistency
- **search()**: K-nearest neighbor search for semantic node retrieval
- **delete()**: Node vector removal with cleanup
- **get()**: Direct node vector record access
- **items()**: Iterator access to all stored vectors

#### Entity Integration:
- **Entity Vectors**: Embedded entity names for semantic search
- **Path Fragment Vectors**: Relationship triplet embeddings
- **Metadata Storage**: Rich payload with entity/path metadata
- **Type Classification**: Entity vs path vector differentiation

#### Use Cases:
- **Semantic Entity Search**: Find similar entities by vector similarity
- **Path Pattern Matching**: Discover similar relationship patterns
- **Graph Navigation**: Vector-guided graph traversal
- **Entity Disambiguation**: Resolve entity mentions via embeddings

---

### 9. Storage Manager (`storage.py`)

**Purpose**: SQLite-based audit trail and history management for memory operations with migration support.

**Architecture Design**:
- **History Tracking**: Complete audit trail of all memory operations
- **Thread-Safe Operations**: Concurrent access with comprehensive locking
- **Schema Migration**: Automatic database schema evolution
- **Event Logging**: Detailed operation history with actor tracking

**Core Components**:

#### Database Schema:
- **history table**: `id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role`
- **UUID Primary Keys**: Unique identifier for each history record
- **Temporal Tracking**: Created and updated timestamps for audit trails
- **Actor Attribution**: User/agent identification for operations

#### Migration System:
- **_migrate_history_table()**: Automatic schema evolution from legacy formats
- **Backward Compatibility**: Handles old group-chat column schemas
- **Data Preservation**: Copies intersecting data during migrations
- **Rollback Safety**: Transaction-based migration with error recovery

#### History Management:
- **add_history()**: Record memory operations with full context
- **get_history()**: Retrieve chronological operation history
- **Event Types**: ADD, UPDATE, DELETE operation tracking
- **Actor Tracking**: User attribution for all operations

#### Transaction Management:
- **ACID Compliance**: Full transactional integrity
- **BEGIN/COMMIT/ROLLBACK**: Proper transaction handling
- **Error Recovery**: Automatic rollback on failures
- **Thread Safety**: Lock-based concurrent access protection

#### Maintenance Operations:
- **reset()**: Complete history table recreation
- **close()**: Proper connection cleanup
- **__del__()**: Automatic resource cleanup

---

### 10. Utility Functions (`utils.py`)

**Purpose**: Core utility functions for timestamp generation, hashing, and text processing.

**Core Functions**:
- **utc_now_iso()**: ISO format UTC timestamp generation for consistent temporal tracking
- **md5_hash()**: MD5 hashing for content fingerprinting and deduplication
- **merge_texts()**: Token-based text merging for memory consolidation with unique token preservation

---

### 11. Configuration System (`local_config.py`)

**Purpose**: Comprehensive configuration management using Pydantic models with YAML file support and path resolution.

**Architecture Design**:
- **Pydantic Models**: Type-safe configuration with validation
- **Hierarchical Configuration**: Nested configs for different system components
- **File-based Configuration**: YAML configuration file support
- **Path Resolution**: Automatic relative path resolution

**Configuration Components**:

#### Model Configurations:
- **LocalLLMConfig**: LLM backend, device, generation parameters
- **LocalEmbedderConfig**: BGE-M3 model path and embedding dimensions
- **EmotionAnalyzerConfig**: RoBERTa emotion analysis configuration
- **LocalVectorStoreConfig**: FAISS index type, distance strategy, dimensions
- **LocalGraphConfig**: SQLite graph database path
- **EntityIndexConfig**: Entity index database configuration
- **LocalNodeVectorConfig**: Node vector index configuration

#### Retrieval & Processing:
- **ScoringWeights**: Weighted scoring parameters for hybrid retrieval
- **RetrievalConfig**: Search thresholds, K values, debug flags
- **RerankerConfig**: BGE-M3 reranker configuration with advanced fusion
- **EntityMergingConfig**: Entity similarity merging thresholds and controls

#### Master Configuration:
- **OfflineMemoryConfig**: Composite configuration containing all subsystems
- **from_file()**: YAML file loading with path resolution
- **Default Factories**: Intelligent default value generation
- **Backward Compatibility**: Legacy configuration key support

---

## LLM Style Learning System

### 12. Style System Core (`style_system.py`)

**Purpose**: Comprehensive style and emotion learning system that adapts LLM responses to user communication patterns.

**Architecture Design**:
- **Dynamic Style Learning**: Real-time adaptation to user linguistic patterns
- **Emotion Profiling**: Multi-dimensional emotion tracking and replication
- **LLM Integration**: Intel ipex-llm with automatic style injection
- **Persistent Learning**: Profile persistence across sessions

**Core Components**:

#### Style Learning Engine:
- **LinguisticAnalyzer Integration**: Extracts communication patterns from user messages
- **EmotionAnalyzer Integration**: RoBERTa-based emotion detection and scoring
- **StyleEmotionProfile**: Comprehensive user style and emotion profile storage
- **Real-time Learning**: Updates profile on every user interaction

#### Profile Management:
- **Profile Persistence**: JSON-based profile storage with versioning
- **System Prompt Generation**: Dynamic system prompt creation from learned profile
- **Demo Learning**: Quick profile initialization with seed messages
- **Status Monitoring**: Profile statistics and blueprint inspection

#### LLM Response Generation:
- **Intel Backend**: ipex-llm AutoModelForCausalLM with 4-bit optimization
- **Qwen Compatibility**: Automatic rotary embedding patching
- **Style Injection**: System prompt-based style enforcement
- **Chat History**: Conversation context management

#### Emotion Enforcement System:
- **Multi-Emotion Mapping**: Proportional emotion expression
- **Style-Emotion Fusion**: Combined style and emotion application
- **Response Validation**: Ensures consistent emotional expression
- **User Mood Independence**: Maintains learned profile regardless of user input

**Key Features**:

#### Learning Pipeline:
1. **User Message Analysis**: Linguistic and emotional feature extraction
2. **Profile Update**: Real-time profile learning from user patterns
3. **System Prompt Generation**: Dynamic prompt creation with learned patterns
4. **Response Generation**: Style-aware LLM response generation
5. **Profile Persistence**: Automatic saving of learned patterns

#### Style Blueprint Generation:
- **Emotional Palette**: Top 15 emotions with percentages
- **Linguistic Patterns**: Extracted communication style rules
- **System Prompt Integration**: Seamless integration with LLM system prompts
- **Hidden Enforcement**: Style application without user awareness

#### Interactive Features:
- **Live Chat Interface**: Real-time style learning during conversation
- **Status Commands**: Profile inspection and statistics
- **Demo Mode**: Quick style learning with predefined messages
- **Session Persistence**: Continuous learning across chat sessions

---

### 13. Linguistic Analyzer (`linguistic_analyzer.py`)

**Purpose**: Advanced feature extraction system for comprehensive communication pattern analysis and style cloning.

**Architecture Design**:
- **Multi-Dimensional Analysis**: Lexical, structural, punctuation, casing, emoji, and behavioral patterns
- **Structured Output**: FeatureReport dataclass with normalized ratios and raw counts
- **Pattern Detection**: Both predefined and dynamic pattern recognition
- **Extensible Framework**: Customizable dictionaries and pattern categories

**Core Components**:

#### Feature Extraction Engine:
- **FeatureReport**: Comprehensive structured output covering all linguistic dimensions
- **Multi-Category Analysis**: Structure, punctuation, casing, emoji, elongation, repetition, lexical, special patterns
- **Normalization**: Ratio-based normalization for consistent feature comparison
- **Example Collection**: Representative examples for human inspection

#### Pattern Categories:

**Structural Analysis**:
- **Sentence Length**: Average words per sentence
- **Question/Exclamation Ratios**: Sentence type distribution
- **Period Usage**: Declarative sentence patterns

**Punctuation Patterns**:
- **Repeated Punctuation**: !!! ??? ... !!?? detection and counting
- **Oxford Comma**: Preference detection with/without patterns
- **Quote Types**: ASCII vs smart quote usage
- **Special Punctuation**: Semicolon, colon, dash, comma frequency

**Casing Behaviors**:
- **Token Case Analysis**: ALL_CAPS, Title, lower, mixed case ratios
- **Sentence Start Patterns**: Capitalization compliance detection
- **Lowercase 'i'**: Personal pronoun capitalization habits
- **Mixed Case Detection**: StudlyCaps and brand-name patterns

**Emoji Usage**:
- **Position Analysis**: Start/middle/end emoji placement patterns
- **Emoji Frequency**: Per-token emoji density
- **Type Distribution**: Specific emoji usage patterns
- **Grapheme Clustering**: Skin tone and variation selector handling

#### Advanced Pattern Detection:

**Lexical Categories**:
- **Predefined Patterns**: Fillers, hedges, slang, affective expressions
- **Dynamic Detection**: User-specific frequent, rare, and signature words
- **Frequency Analysis**: Word repetition and usage patterns
- **Technical Terms**: Academic/technical vocabulary detection

**User-Specific Patterns**:
- **Frequent Words**: Within-text repetition patterns
- **Signature Words**: Distinctive vocabulary (long words, unusual combinations)
- **Rare Words**: Linguistically uncommon terms detection
- **Technical Vocabulary**: Specialized domain language

**Behavioral Markers**:
- **Elongations**: Character repetition patterns (sooooo, hmmm)
- **Keysmash**: Random character sequences detection
- **Word Repetition**: Adjacent repeated word patterns
- **URL/Handle Usage**: Social media pattern detection

#### Dynamic Pattern Discovery:

**Conservative Detection Strategies**:
- **Slang Detection**: Internet abbreviations and modern terms
- **Affective Recognition**: Emotional interjections and sounds
- **Accuracy Priority**: Prefers precision over recall to avoid false positives
- **Context Awareness**: Pattern validation within linguistic context

**N-gram Analysis**:
- **Bigram/Trigram Extraction**: Common phrase patterns
- **Canonical Tokenization**: Normalized token sequences
- **Phrase Pattern Recognition**: Recurring multi-word expressions

#### Processing Pipeline:

**Text Normalization**:
- **Space Normalization**: Whitespace and Unicode punctuation handling
- **Diacritic Removal**: Character normalization for comparison
- **Elongation Collapse**: Character repetition standardization
- **Token Canonicalization**: Consistent token representation

**Feature Computation**:
- **Ratio Calculation**: Normalized frequency measurements
- **Example Collection**: Representative pattern instances
- **Count Aggregation**: Raw frequency statistics
- **Statistical Summaries**: Mean, ratio, and distribution calculations

#### Integration Features:

**Compatibility Layer**:
- **extract_patterns_from_text()**: Legacy API compatibility
- **extract_phrase_counts_from_text()**: FAISS integration support
- **Report Merging**: Batch processing and aggregation
- **Structured Output**: JSON-serializable results

**Performance Optimizations**:
- **Compiled Regex**: Pre-compiled pattern matching
- **Unicode Support**: Full international text handling
- **Memory Efficiency**: Bounded example collection
- **Batch Processing**: Multi-report aggregation support

---

### 14. Pattern Deduplication Engine (`pattern_deduplicator.py`)

**Purpose**: Advanced multi-layer pattern clustering system for intelligent deduplication and canonicalization of linguistic patterns.

**Architecture Design**:
- **Multi-Layer Clustering**: Exact matching, edit distance, phonetic similarity, and semantic clustering
- **Incremental Learning**: Persistent memory system for pattern recognition
- **Category-Specific Strategies**: Specialized clustering algorithms for different pattern types
- **FAISS Integration**: Semantic similarity clustering for complex patterns

**Core Components**:

#### Clustering Strategies:

**Layer 1 - Memory Lookup**: 
- **Fastest Path**: Check existing canonical forms from previous learning
- **Instant Recognition**: Direct mapping for previously seen patterns
- **Incremental Updates**: Add to existing clusters with count accumulation

**Layer 2 - Category-Specific Clustering**:
- **Elongation Clustering**: Groups variants like "sooo", "soooo", "sooooo"
- **Phonetic Clustering**: Groups filler variants like "um", "umm", "ummm"
- **Edit Distance Clustering**: Groups typographic variants within threshold
- **Semantic Clustering**: FAISS-based grouping for signature/rare words

#### Data Structures:

**ClusterNode**:
- **Canonical Form**: Representative word for cluster
- **Variant Set**: All variations within cluster
- **Count Tracking**: Total frequency across all variants
- **Category Classification**: Pattern type assignment

**PatternMemory**:
- **Cluster Storage**: Canonical form to cluster mapping
- **Word Mapping**: Individual word to canonical form lookup
- **Category Statistics**: Aggregated counts per pattern category
- **Persistence**: JSON serialization for session continuity

#### Similarity Algorithms:

**Edit Distance Calculation**:
- **Levenshtein Distance**: Character-level difference measurement
- **Threshold-Based Clustering**: Configurable similarity requirements
- **Shortest Canonical**: Prefers shorter forms as representatives

**Phonetic Similarity**:
- **Character Collapse**: Removes repeated characters for comparison
- **Elongation Tolerance**: Handles filler word variations
- **Subset Detection**: Recognizes partial matches in sound patterns

**Elongation Detection**:
- **Character Repetition**: Identifies repeated character patterns
- **Variant Recognition**: Groups elongated forms of same base
- **Canonical Selection**: Chooses shortest valid form

#### Semantic Clustering:

**FAISS Integration**:
- **Vector Similarity**: Embedding-based semantic comparison
- **High Threshold**: Conservative clustering (0.85+ similarity)
- **Dynamic Addition**: Adds new vectors for novel patterns
- **Contextual Grouping**: Groups semantically related terms

**Category-Specific Processing**:
- **Signature Words**: Technical vocabulary clustering
- **Rare Words**: Unusual term grouping
- **Frequent Words**: High-frequency pattern recognition

#### Pattern Categories:

**Elongations**: Character repetition patterns
**Fillers**: Vocal pause and hesitation markers
**Hedges**: Uncertainty and qualification expressions
**Signature Words**: User-specific distinctive vocabulary
**Rare Words**: Linguistically uncommon terms
**Frequent Words**: High-repetition user patterns

#### Memory Management:

**Persistence System**:
- **JSON Storage**: Human-readable memory serialization
- **Load/Save Operations**: Session continuity across restarts
- **Incremental Updates**: Continuous learning without data loss

**Statistics Tracking**:
- **Category Totals**: Aggregate pattern frequency
- **Percentage Calculation**: Relative pattern importance
- **Top-N Reporting**: Most significant patterns per category

#### Performance Features:

**Processing Efficiency**:
- **Layer Prioritization**: Fast paths for common operations
- **Threshold Optimization**: Tunable similarity requirements
- **Memory Caching**: Avoided duplicate calculations

**Output Formatting**:
- **Percentage Display**: Pattern importance visualization
- **Count Information**: Frequency data for analysis
- **Canonical Forms**: Standardized pattern representation
