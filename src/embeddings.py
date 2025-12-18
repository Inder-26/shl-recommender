"""
Embedding Generator and ChromaDB Vector Store
Creates embeddings for SHL assessments and stores in ChromaDB
"""

import json
import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


class AssessmentVectorStore:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 db_path: str = "data/chroma_db"):
        """
        Initialize the vector store with embedding model and ChromaDB
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"Initializing ChromaDB at: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="shl_assessments",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store ready. Collection has {self.collection.count()} items.")
    
    def create_document_text(self, assessment: dict) -> str:
        """
        Create rich text representation for embedding
        This is critical for good search results
        """
        parts = []
        
        # Name
        name = assessment.get('name', '')
        parts.append(f"Assessment Name: {name}")
        
        # Test Type - expand codes to full descriptions
        test_type = assessment.get('test_type', '')
        type_descriptions = {
            'A': 'Ability and Aptitude Test',
            'B': 'Biodata and Situational Judgment',
            'C': 'Competency Assessment',
            'D': 'Development Assessment',
            'E': 'Exercises and Simulations',
            'K': 'Knowledge and Skills Test Technical',
            'P': 'Personality and Behavioral Assessment',
            'S': 'Simulation'
        }
        
        expanded_types = []
        for char in str(test_type):
            if char in type_descriptions:
                expanded_types.append(type_descriptions[char])
        
        if expanded_types:
            parts.append(f"Test Types: {', '.join(expanded_types)}")
        
        # Remote and Adaptive
        if assessment.get('remote_testing'):
            parts.append("Supports Remote Testing")
        if assessment.get('adaptive_irt'):
            parts.append("Supports Adaptive IRT Testing")
        
        # Description
        if assessment.get('description'):
            parts.append(f"Description: {assessment['description']}")
        
        # Keywords from name
        keywords = self._extract_keywords(name)
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        
        return " | ".join(parts)
    
    def _extract_keywords(self, name: str) -> list:
        """Extract relevant keywords from assessment name"""
        if not name:
            return []
            
        name_lower = name.lower()
        keywords = []
        
        # Technical skills mapping
        tech_keywords = {
            'java': ['java', 'programming', 'developer', 'software', 'backend', 'jvm'],
            'python': ['python', 'programming', 'developer', 'software', 'scripting', 'data'],
            'sql': ['sql', 'database', 'data', 'query', 'relational'],
            '.net': ['dotnet', 'microsoft', 'developer', 'csharp', 'asp'],
            'javascript': ['javascript', 'js', 'frontend', 'web', 'react', 'angular', 'node'],
            'c++': ['cpp', 'programming', 'systems', 'embedded'],
            'c#': ['csharp', 'dotnet', 'microsoft', 'unity'],
            'excel': ['excel', 'spreadsheet', 'microsoft', 'office', 'data', 'analysis'],
            'salesforce': ['salesforce', 'crm', 'sales', 'cloud', 'customer'],
            'sap': ['sap', 'erp', 'enterprise', 'business'],
            'html': ['html', 'web', 'frontend', 'markup'],
            'css': ['css', 'web', 'frontend', 'design', 'styling'],
            'angular': ['angular', 'frontend', 'javascript', 'web', 'typescript'],
            'react': ['react', 'frontend', 'javascript', 'web', 'redux'],
            'php': ['php', 'web', 'backend', 'laravel'],
            'ruby': ['ruby', 'rails', 'web', 'backend'],
            'scala': ['scala', 'jvm', 'functional', 'spark'],
            'kotlin': ['kotlin', 'android', 'jvm', 'mobile'],
            'swift': ['swift', 'ios', 'apple', 'mobile'],
            'android': ['android', 'mobile', 'java', 'kotlin', 'app'],
            'ios': ['ios', 'mobile', 'swift', 'apple', 'iphone'],
            'aws': ['aws', 'amazon', 'cloud', 'devops'],
            'azure': ['azure', 'microsoft', 'cloud', 'devops'],
            'docker': ['docker', 'container', 'devops', 'kubernetes'],
        }
        
        for tech, related in tech_keywords.items():
            if tech in name_lower:
                keywords.extend(related)
        
        # Role and soft skill keywords
        role_keywords = {
            'manager': ['management', 'leadership', 'supervisor', 'team lead', 'director'],
            'analyst': ['analysis', 'analytical', 'data', 'business', 'reporting'],
            'developer': ['development', 'programming', 'software', 'coding', 'engineer'],
            'administrator': ['administration', 'admin', 'office', 'clerical'],
            'sales': ['selling', 'customer', 'revenue', 'business development', 'account'],
            'customer service': ['customer', 'service', 'support', 'helpdesk', 'call center'],
            'verbal': ['verbal', 'communication', 'language', 'english', 'reading'],
            'numerical': ['numerical', 'math', 'quantitative', 'numbers', 'calculation'],
            'reasoning': ['reasoning', 'logical', 'problem solving', 'critical thinking'],
            'personality': ['personality', 'behavior', 'soft skills', 'traits', 'character'],
            'leadership': ['leadership', 'management', 'team', 'influence', 'executive'],
            'communication': ['communication', 'interpersonal', 'collaboration', 'verbal'],
            'cognitive': ['cognitive', 'mental', 'thinking', 'aptitude', 'ability'],
            'attention': ['attention', 'detail', 'accuracy', 'focus', 'concentration'],
            'clerical': ['clerical', 'office', 'administrative', 'data entry'],
            'mechanical': ['mechanical', 'technical', 'engineering', 'physical'],
            'spatial': ['spatial', 'visual', 'geometric', 'perception'],
        }
        
        for role, related in role_keywords.items():
            if role in name_lower:
                keywords.extend(related)
        
        return list(set(keywords))
    
    def load_assessments(self, json_path: str) -> list:
        """Load assessments from JSON file"""
        print(f"Loading assessments from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
        print(f"Loaded {len(assessments)} assessments")
        return assessments
    
    def index_assessments(self, assessments: list):
        """Index all assessments into ChromaDB"""
        print(f"Indexing {len(assessments)} assessments...")
        
        # Clear existing collection
        try:
            self.client.delete_collection("shl_assessments")
            print("  Cleared existing collection")
        except:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name="shl_assessments",
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        print("  Creating embeddings...")
        for i, assessment in enumerate(tqdm(assessments, desc="  Embedding")):
            doc_text = self.create_document_text(assessment)
            embedding = self.model.encode(doc_text).tolist()
            
            # Parse duration
            duration = 30  # default
            dur_val = assessment.get('duration', '')
            if dur_val:
                match = re.search(r'\d+', str(dur_val))
                if match:
                    duration = int(match.group())
            
            metadata = {
                'name': str(assessment.get('name', '')),
                'url': str(assessment.get('url', '')),
                'test_type': str(assessment.get('test_type', '')),
                'duration': str(duration),
                'remote_testing': str(assessment.get('remote_testing', False)),
                'adaptive_irt': str(assessment.get('adaptive_irt', False)),
                'description': str(assessment.get('description', ''))[:500],
            }
            
            ids.append(f"assessment_{i}")
            documents.append(doc_text)
            metadatas.append(metadata)
            embeddings.append(embedding)
        
        # Add in batches
        print("  Adding to ChromaDB...")
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="  Storing"):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end]
            )
        
        print(f"Indexed {len(ids)} assessments into ChromaDB")
    
    def search(self, query: str, n_results: int = 10) -> list:
        """
        Hybrid Search: Vector Similarity + Keyword Boosting
        Returns deduplicated, re-ranked results
        """
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Fetch more results for re-ranking
        fetch_count = min(n_results * 5, 100)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'][0]:
            return []
        
        # Extract keywords from query for boosting
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        
        # Important keywords that should boost heavily
        boost_keywords = {
            'java', 'python', 'sql', 'javascript', 'excel', 'sap', 'salesforce',
            'verbal', 'numerical', 'cognitive', 'personality', 'leadership',
            'sales', 'customer', 'service', 'english', 'communication',
            'analyst', 'developer', 'manager', 'administrative', 'clerical',
            'reasoning', 'aptitude', 'behavioral', 'mechanical', 'spatial'
        }
        
        query_boost_words = query_words.intersection(boost_keywords)
        
        # Process and score results
        formatted = []
        seen_urls = set()
        
        type_mapping = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgment',
            'C': 'Competency',
            'D': 'Development',
            'E': 'Exercises',
            'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior',
            'S': 'Simulation'
        }
        
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            url = meta.get('url', '')
            
            # Normalize URL for deduplication
            norm_url = url.strip().lower().rstrip('/')
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)
            
            # Calculate base score (1 - distance)
            base_score = 1 - results['distances'][0][i]
            
            # Keyword boosting
            name_lower = meta.get('name', '').lower()
            boost = 0.0
            
            for word in query_boost_words:
                if word in name_lower:
                    boost += 0.3  # Significant boost for exact match
            
            # Additional boost for partial matches
            for word in query_words:
                if len(word) > 3 and word in name_lower:
                    boost += 0.1
            
            final_score = min(base_score + boost, 1.0)
            
            # Parse test type
            test_type_raw = meta.get('test_type', '')
            test_types = []
            for char in str(test_type_raw):
                if char in type_mapping:
                    test_types.append(type_mapping[char])
            if not test_types:
                test_types = ['General']
            
            # Parse duration
            duration = 30
            try:
                duration = int(meta.get('duration', '30'))
            except:
                pass
            
            formatted.append({
                'name': meta.get('name', ''),
                'url': url,
                'description': meta.get('description', ''),
                'test_type': test_types,
                'test_type_raw': test_type_raw,
                'duration': duration,
                'remote_testing': meta.get('remote_testing', 'False') == 'True',
                'adaptive_irt': meta.get('adaptive_irt', 'False') == 'True',
                'score': round(final_score, 4),
                'boosted': boost > 0
            })
        
        # Sort by final score
        formatted.sort(key=lambda x: x['score'], reverse=True)
        
        return formatted[:n_results]
    
    def search_by_type(self, query: str, test_type: str, n_results: int = 10) -> list:
        """Search filtered by test type (K, P, A, etc.)"""
        all_results = self.search(query, n_results=50)
        filtered = [r for r in all_results if test_type in r.get('test_type_raw', '')]
        return filtered[:n_results]


def build_index():
    """Build the vector index from scraped data"""
    json_path = 'data/raw/shl_catalog.json'
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        print("Run scraper first: python src/scraper.py")
        return None
    
    store = AssessmentVectorStore()
    assessments = store.load_assessments(json_path)
    store.index_assessments(assessments)
    
    print("\nIndex built successfully!")
    print(f"Total assessments: {store.collection.count()}")
    
    return store


def test_search():
    """Test the search functionality"""
    store = AssessmentVectorStore()
    
    test_queries = [
        "Java developer with team collaboration skills",
        "Python SQL data analyst",
        "Customer service personality assessment",
        "Cognitive aptitude test",
        "Sales manager leadership"
    ]
    
    print("\n" + "="*60)
    print("SEARCH TEST")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = store.search(query, n_results=5)
        for i, r in enumerate(results, 1):
            boosted = " [BOOSTED]" if r.get('boosted') else ""
            print(f"  {i}. {r['name']} (score: {r['score']:.3f}){boosted}")
            print(f"     Type: {r['test_type_raw']} | Duration: {r['duration']}min")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_search()
    else:
        build_index()