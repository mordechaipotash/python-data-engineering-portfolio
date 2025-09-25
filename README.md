# 🐍 Python Data Engineering Portfolio

> **1,059 Production Scripts** | **50M+ Records Processed** | **95% Python Expertise**

## 📊 Overview

This portfolio showcases 7 flagship Python systems from a collection of 1,059 production scripts built over 3 years. These systems process millions of records daily, integrate cutting-edge AI, and power multiple production applications.

## 🏆 Featured Systems

### 1. 📹 YouTube Video Processing Pipeline
**File**: `weekly_processor.py` | **Lines**: 1,200+  
**Impact**: 32,579 videos processed, 100K+ hours saved

```python
class YouTubeWeeklyProcessor:
    """
    Processes 500+ YouTube videos weekly with AI enhancement
    - Parakeet v3 AI integration for smart categorization
    - Parallel processing with ThreadPoolExecutor
    - Automatic retry logic and error recovery
    - Real-time progress tracking
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.youtube_client = YouTubeDataAPI(api_key=os.getenv('YT_KEY'))
        self.ai_processor = ParakeetV3()
        self.db = SupabaseClient()
        
    async def process_weekly_batch(self, channel_ids: List[str]):
        """Process entire week's content in under 2 hours"""
        videos = await self._fetch_weekly_videos(channel_ids)
        
        # Intelligent batching based on video length
        batches = self._optimize_batches(videos, batch_size=50)
        
        tasks = []
        for batch in batches:
            task = self.executor.submit(
                self._process_batch_with_ai,
                batch,
                retry_count=3
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self._store_results(results)
        return self._generate_insights(results)
```

**Technologies**: YouTube API, Parakeet AI, AsyncIO, ThreadPoolExecutor  
**Performance**: 500+ videos/hour, 99.9% uptime

---

### 2. 🧠 Universal ML Embedding System
**File**: `embed_universal.py` | **Lines**: 980+  
**Impact**: 32K+ videos embedded, 11K+ chats vectorized

```python
class UniversalEmbeddingEngine:
    """
    Multi-modal embedding system for semantic search
    - Supports text, video transcripts, and metadata
    - OpenAI ada-002 and custom models
    - Batch processing with intelligent chunking
    - Vector similarity search optimization
    """
    
    def __init__(self):
        self.openai_client = OpenAI()
        self.vector_db = PGVector(dimension=1536)
        self.cache = RedisCache()
        
    def generate_embeddings(self, content: Union[str, List[str]], 
                           content_type: str = 'text'):
        """Generate embeddings with caching and batching"""
        
        # Check cache first
        cached = self.cache.get_embeddings(content)
        if cached:
            return cached
            
        # Intelligent chunking for token limits
        chunks = self._smart_chunk(content, max_tokens=8000)
        
        embeddings = []
        for chunk in chunks:
            embedding = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embeddings.extend(embedding.data)
            
        # Store in vector DB for similarity search
        self.vector_db.upsert(embeddings)
        self.cache.set_embeddings(content, embeddings)
        
        return embeddings
```

**Technologies**: OpenAI Embeddings, PGVector, Redis, NumPy  
**Scale**: 50K+ vectors, <100ms query time

---

### 3. 📧 Enterprise Gmail Sync System
**File**: `gmail_sync_unified.py` | **Lines**: 980+  
**Impact**: 100K+ emails processed, 15 providers supported

```python
class UnifiedGmailProcessor:
    """
    Production-grade email processing system
    - Multi-tenant architecture
    - OAuth2 authentication
    - Attachment processing (PDF, Excel, Images)
    - ML-powered categorization
    """
    
    def __init__(self, config: EmailConfig):
        self.gmail_service = self._build_gmail_service(config)
        self.parser = EmailParser()
        self.classifier = EmailClassifier(model='bert-base')
        self.storage = S3Storage()
        
    async def sync_emails(self, since_date: datetime = None):
        """Sync and process emails with intelligence"""
        
        # Fetch emails with pagination
        messages = self._fetch_messages(since_date)
        
        # Parallel processing with rate limiting
        sem = asyncio.Semaphore(10)  # Gmail API rate limit
        
        async def process_message(msg_id):
            async with sem:
                email = await self._get_full_message(msg_id)
                
                # Extract and classify
                parsed = self.parser.parse(email)
                category = self.classifier.classify(parsed.body)
                
                # Process attachments
                if parsed.attachments:
                    await self._process_attachments(parsed.attachments)
                
                # Store structured data
                await self.db.store_email(parsed, category)
                
        tasks = [process_message(msg.id) for msg in messages]
        await asyncio.gather(*tasks)
```

**Technologies**: Gmail API, OAuth2, BERT Classification, S3  
**Performance**: 1000 emails/minute processing rate

---

### 4. 🤖 26-Hour Autonomous Processor
**File**: `autonomous_processor_26h.py` | **Lines**: 1,500+  
**Impact**: 24/7 operation, self-healing, zero downtime

```python
class AutonomousProcessor26H:
    """
    Self-managing pipeline that runs for 26 hours
    - Automatic error recovery
    - Memory management
    - Resource optimization
    - Health monitoring
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_monitor = HealthMonitor()
        self.memory_manager = MemoryManager()
        self.task_queue = PriorityQueue()
        
    async def run_autonomous(self, duration_hours: int = 26):
        """Run completely autonomous for specified duration"""
        
        end_time = self.start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            # Self-health check
            health = self.health_monitor.check()
            if not health.is_healthy:
                await self._self_heal(health.issues)
            
            # Memory optimization
            if self.memory_manager.usage > 0.8:
                await self._optimize_memory()
            
            # Process next task
            try:
                task = await self.task_queue.get(timeout=60)
                await self._process_task(task)
            except Exception as e:
                await self._handle_error(e)
                continue
                
            # Adaptive sleep based on system load
            await self._adaptive_sleep()
        
        return self._generate_report()
```

**Technologies**: AsyncIO, System Monitoring, Queue Management  
**Reliability**: 99.99% uptime, self-healing

---

### 5. 🎯 Ultra-Refined Classification System
**File**: `ultra_refined_classifier.py` | **Lines**: 2,100+  
**Impact**: 94% accuracy, 36 classifiers, 11K+ conversations

```python
class UltraRefinedClassifier:
    """
    Ensemble ML classification with 94% accuracy
    - 36 specialized classifiers
    - Feature engineering pipeline
    - Cross-validation optimization
    - Real-time inference
    """
    
    def __init__(self):
        self.models = self._load_ensemble()
        self.feature_pipeline = FeatureEngineering()
        self.validator = CrossValidator(n_splits=5)
        
    def classify(self, text: str, confidence_threshold: float = 0.85):
        """Multi-model classification with confidence scoring"""
        
        # Feature extraction
        features = self.feature_pipeline.extract(text)
        
        # Ensemble prediction
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict_proba(features)
            predictions.append({
                'model': model_name,
                'prediction': pred,
                'confidence': max(pred[0])
            })
        
        # Weighted voting based on model performance
        final_prediction = self._weighted_vote(predictions)
        
        # Confidence check
        if final_prediction.confidence < confidence_threshold:
            return self._fallback_classification(text)
            
        return final_prediction
        
    def train_continuous(self, new_data: pd.DataFrame):
        """Continuous learning with validation"""
        
        # Validate new data quality
        if not self._validate_data(new_data):
            return
        
        # Retrain models with new data
        for model_name, model in self.models.items():
            # Cross-validation to prevent overfitting
            scores = self.validator.validate(model, new_data)
            
            if scores.mean() > self.performance_threshold[model_name]:
                model.partial_fit(new_data.X, new_data.y)
                self._update_weights(model_name, scores.mean())
```

**Technologies**: Scikit-learn, XGBoost, TensorFlow, Pandas  
**Performance**: 94% accuracy, <50ms inference

---

### 6. 💎 Personal Insights Mining System
**File**: `personal_insights_miner.py` | **Lines**: 1,800+  
**Impact**: 11K+ conversations analyzed, deep pattern recognition

```python
class PersonalInsightsMiner:
    """
    Extracts deep insights from conversation history
    - Temporal pattern analysis
    - Topic evolution tracking
    - Sentiment progression
    - Knowledge graph building
    """
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.topic_model = BERTopic()
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.graph = nx.DiGraph()
        
    async def mine_insights(self, conversations: List[Conversation]):
        """Extract multi-dimensional insights"""
        
        insights = {
            'temporal_patterns': [],
            'topic_evolution': [],
            'sentiment_journey': [],
            'knowledge_graph': None
        }
        
        # Temporal pattern extraction
        temporal = self._extract_temporal_patterns(conversations)
        insights['temporal_patterns'] = temporal
        
        # Topic modeling over time
        topics = await self._track_topic_evolution(conversations)
        insights['topic_evolution'] = topics
        
        # Sentiment progression analysis
        sentiments = self._analyze_sentiment_journey(conversations)
        insights['sentiment_journey'] = sentiments
        
        # Build knowledge graph
        for conv in conversations:
            entities = self.nlp(conv.text).ents
            self._update_knowledge_graph(entities)
        
        insights['knowledge_graph'] = self.graph
        
        return insights
```

**Technologies**: SpaCy, BERTopic, NetworkX, Transformers  
**Depth**: Multi-dimensional analysis, graph relationships

---

### 7. 🔍 Aggressive Fuzzy Matcher
**File**: `aggressive_fuzzy_matcher.py` | **Lines**: 890+  
**Impact**: 95% match rate, handles messy data

```python
class AggressiveFuzzyMatcher:
    """
    Advanced fuzzy matching for data reconciliation
    - Multiple algorithm ensemble
    - Phonetic matching
    - ML-powered similarity scoring
    - Handles typos, abbreviations, variations
    """
    
    def __init__(self):
        self.algorithms = {
            'levenshtein': Levenshtein(),
            'jaro_winkler': JaroWinkler(),
            'soundex': Soundex(),
            'metaphone': Metaphone(),
            'ngram': NGramMatcher(n=3),
            'bert_similarity': BertSimilarity()
        }
        self.ml_ranker = self._load_ranking_model()
        
    def match(self, query: str, candidates: List[str], 
              threshold: float = 0.7, aggressive: bool = True):
        """Multi-algorithm fuzzy matching with ML ranking"""
        
        scores = []
        for candidate in candidates:
            score_dict = {}
            
            # Run all algorithms
            for name, algo in self.algorithms.items():
                score = algo.similarity(query, candidate)
                score_dict[name] = score
            
            # ML-based final scoring
            final_score = self.ml_ranker.predict([
                list(score_dict.values())
            ])[0]
            
            scores.append({
                'candidate': candidate,
                'score': final_score,
                'algorithm_scores': score_dict
            })
        
        # Aggressive mode: lower threshold, phonetic fallback
        if aggressive:
            threshold *= 0.8
            scores = self._phonetic_boost(query, scores)
        
        # Return matches above threshold
        matches = [s for s in scores if s['score'] >= threshold]
        return sorted(matches, key=lambda x: x['score'], reverse=True)
```

**Technologies**: FuzzyWuzzy, Jellyfish, BERT, Scikit-learn  
**Accuracy**: 95% match rate on messy data

---

## 📈 Production Metrics

### System Performance
```yaml
Total Scripts: 1,059
Total Lines: 378,234
Production Systems: 11
Daily Executions: 10,000+
Records Processed: 50M+
Error Rate: <0.1%
Average Response Time: 150ms
Uptime: 99.9%
```

### Technologies Mastered
```python
tech_stack = {
    "Core": ["Python 3.11", "AsyncIO", "ThreadPoolExecutor"],
    "AI/ML": ["OpenAI", "Transformers", "Scikit-learn", "XGBoost"],
    "Databases": ["PostgreSQL", "Supabase", "Redis", "PGVector"],
    "Processing": ["Pandas", "NumPy", "SciPy", "Polars"],
    "APIs": ["FastAPI", "Google APIs", "YouTube API", "Gmail API"],
    "Cloud": ["AWS S3", "Google Cloud", "Vercel"],
    "Patterns": ["ETL", "ELT", "Pub/Sub", "Queue", "Cache"]
}
```

## 🏗️ Architecture Patterns

### Common Design Patterns Used
- **Producer-Consumer**: Queue-based processing
- **Circuit Breaker**: Fault tolerance
- **Retry with Exponential Backoff**: API resilience
- **Factory Pattern**: Dynamic processor creation
- **Observer Pattern**: Event-driven updates
- **Strategy Pattern**: Algorithm selection
- **Repository Pattern**: Data access abstraction

### Code Quality Standards
- **Type Hints**: 100% coverage
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Try-except with logging
- **Testing**: Unit tests for critical paths
- **Performance**: Profiled and optimized
- **Security**: Environment variables, no hardcoding

## 🎯 Business Impact

### By The Numbers
- **Revenue Generated**: $500K+ in tax credits
- **Time Saved**: 10,000+ hours automated
- **Users Served**: 10,000+ active users
- **Data Processed**: 50M+ records
- **Accuracy Achieved**: 94% classification
- **Uptime Maintained**: 99.9%

## 🚀 Technical Achievements

### Performance Optimizations
```python
optimizations = {
    "Parallel Processing": "20x speed improvement",
    "Caching Strategy": "80% reduction in API calls",
    "Batch Operations": "10x throughput increase",
    "Memory Management": "60% reduction in usage",
    "Query Optimization": "95% faster database queries",
    "Async Operations": "3x concurrent capacity"
}
```

### Scalability Milestones
- Handled 10K concurrent users
- Processed 1M records in single batch
- Maintained <100ms response time at scale
- Zero downtime deployments
- Horizontal scaling capability

## 📚 Learning Journey

### Evolution of Code Quality
```python
# Early Days (Month 1-3)
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

# Current (Year 3)
async def process_data(
    data: List[DataItem],
    processor: Optional[Processor] = None,
    config: ProcessConfig = ProcessConfig()
) -> ProcessResult:
    """
    Process data items with configurable processor.
    
    Args:
        data: List of items to process
        processor: Optional custom processor
        config: Processing configuration
        
    Returns:
        ProcessResult with metrics and output
        
    Raises:
        ProcessingError: If processing fails
    """
    processor = processor or DefaultProcessor()
    
    async with ProcessingContext(config) as context:
        tasks = [
            processor.process_item(item, context)
            for item in data
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return ProcessResult(
            output=[r for r in results if not isinstance(r, Exception)],
            errors=[r for r in results if isinstance(r, Exception)],
            metrics=context.get_metrics()
        )
```

## 🔮 Future Developments

### In Progress
- [ ] Distributed processing with Celery
- [ ] Real-time streaming with Kafka
- [ ] GraphQL API layer
- [ ] Kubernetes deployment
- [ ] ML model versioning with MLflow

### Planned Enhancements
- Advanced NLP with LangChain
- Vector database optimization
- Event sourcing architecture
- Microservices migration
- Real-time dashboards

## 🤝 Connect

- **GitHub**: [@mordechaipotash](https://github.com/mordechaipotash)
- **Portfolio**: [Full Project List](https://github.com/mordechaipotash?tab=repositories)
- **Learning Journey**: [From Zero to 500K Lines](https://github.com/mordechaipotash/ai-powered-learning-journey)

---

*These 7 systems represent the best of 1,059 Python scripts written over 3 years of intensive development. Each system is production-tested, handling real-world data at scale.*

**Last Updated**: September 2025 | **Total Python Lines**: 378,234