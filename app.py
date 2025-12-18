"""
SHL Assessment Recommendation API
With Groq LLM integration and reliable fallback
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
from src.embeddings import AssessmentVectorStore

# Load environment variables
load_dotenv()

# Suppress telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = Flask(__name__)
CORS(app)

# Global variables
store = None
groq_client = None


def initialize():
    """Initialize vector store and Groq client"""
    global store, groq_client
    
    print("Initializing SHL Recommendation System...")
    
    # Check if embeddings need to be built
    chroma_path = "data/chroma_db"
    json_path = "data/raw/shl_catalog.json"
    
    # Initialize vector store
    try:
        from src.embeddings import AssessmentVectorStore
        store = AssessmentVectorStore()
        
        # Check if collection is empty
        if store.collection.count() == 0:
            print("  ChromaDB is empty. Building embeddings...")
            if os.path.exists(json_path):
                assessments = store.load_assessments(json_path)
                store.index_assessments(assessments)
                print(f"  Built embeddings for {store.collection.count()} assessments")
            else:
                print(f"  ERROR: {json_path} not found!")
                raise FileNotFoundError(f"Data file not found: {json_path}")
        else:
            print(f"  Vector store ready with {store.collection.count()} assessments")
    except Exception as e:
        print(f"  Vector store initialization failed: {e}")
        raise
    
    # Initialize Groq client
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_api_key)
            # Test the client
            test_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            print("  Groq LLM initialized successfully")
        except Exception as e:
            print(f"  Groq initialization failed: {e}")
            groq_client = None
    else:
        print("  No GROQ_API_KEY found, using keyword fallback")
        groq_client = None
    
    print("System ready!")


def is_url(text: str) -> bool:
    """Check if input is a URL"""
    try:
        text = text.strip()
        result = urlparse(text)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False


def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url.strip(), headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:5000]
    except Exception as e:
        print(f"  URL extraction failed: {e}")
        return ""


def normalize_url(url: str) -> str:
    """Normalize URL for consistent comparison"""
    if not url:
        return ""
    url = str(url).strip().lower().rstrip('/')
    if '/products/product-catalog/' in url and '/solutions/products/' not in url:
        url = url.replace('/products/product-catalog/', '/solutions/products/product-catalog/')
    return url


def analyze_with_groq(query: str) -> dict:
    """Use Groq LLM to analyze query and extract requirements"""
    global groq_client
    
    if groq_client is None:
        return None
    
    try:
        prompt = f"""Analyze this job requirement query and extract key information for finding relevant assessments.

Query:
{query[:2000]}

Return ONLY a valid JSON object with these fields:
{{
    "technical_skills": ["list of technical skills mentioned like Python, Java, SQL, Excel, etc."],
    "soft_skills": ["list of soft skills like communication, leadership, teamwork, etc."],
    "role_type": "one of: technical, analyst, sales, customer_service, managerial, administrative, content, finance, general",
    "experience_level": "one of: entry, mid, senior, any",
    "duration_max_minutes": null or integer if time limit mentioned,
    "needs_cognitive_test": true or false,
    "needs_personality_test": true or false,
    "key_search_terms": ["5-10 most important terms to search for assessments"],
    "enhanced_query": "a simplified 20-30 word search query focusing on key skills and role"
}}

Return ONLY the JSON, no explanation or markdown."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if '```' in result_text:
            result_text = re.sub(r'```json?\s*', '', result_text)
            result_text = re.sub(r'\s*```', '', result_text)
        
        result = json.loads(result_text)
        print(f"  Groq analysis: role={result.get('role_type')}, " +
              f"tech={len(result.get('technical_skills', []))}, " +
              f"soft={len(result.get('soft_skills', []))}")
        
        return result
    
    except Exception as e:
        print(f"  Groq analysis failed: {str(e)[:50]}")
        return None


def analyze_with_keywords(query: str) -> dict:
    """Fallback keyword-based analysis"""
    query_lower = query.lower()
    
    # Technical skills
    tech_keywords = {
        'python', 'java', 'javascript', 'sql', 'c++', 'c#', '.net',
        'excel', 'sap', 'salesforce', 'aws', 'azure', 'react', 'angular',
        'html', 'css', 'php', 'ruby', 'scala', 'kotlin', 'swift',
        'programming', 'coding', 'developer', 'software', 'technical',
        'data', 'database', 'api', 'backend', 'frontend', 'fullstack',
        'machine learning', 'ai', 'devops', 'cloud', 'agile', 'scrum'
    }
    
    # Soft skills
    soft_keywords = {
        'communication', 'leadership', 'management', 'teamwork', 'collaboration',
        'sales', 'customer', 'service', 'personality', 'behavioral', 'interpersonal',
        'negotiation', 'presentation', 'influence', 'motivation', 'creative',
        'problem solving', 'decision making', 'conflict resolution'
    }
    
    # Cognitive keywords
    cognitive_keywords = {
        'cognitive', 'aptitude', 'reasoning', 'numerical', 'verbal', 'logical',
        'analytical', 'problem solving', 'critical thinking', 'ability', 'mental'
    }
    
    tech_found = [k for k in tech_keywords if k in query_lower]
    soft_found = [k for k in soft_keywords if k in query_lower]
    needs_cognitive = any(k in query_lower for k in cognitive_keywords)
    needs_personality = any(k in query_lower for k in ['personality', 'behavioral', 'behaviour'])
    
    # Duration extraction
    duration_limit = None
    duration_patterns = [
        r'(\d+)\s*(?:min|minute|mins)',
        r'max(?:imum)?\s*(?:of\s*)?(\d+)',
        r'under\s*(\d+)',
        r'less than\s*(\d+)',
        r'within\s*(\d+)'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, query_lower)
        if match:
            duration_limit = int(match.group(1))
            break
    
    # Role detection
    role_type = 'general'
    role_patterns = [
        ('developer', 'technical'), ('engineer', 'technical'), ('programmer', 'technical'),
        ('analyst', 'analyst'), ('data', 'analyst'),
        ('manager', 'managerial'), ('director', 'managerial'), ('executive', 'managerial'),
        ('sales', 'sales'), ('account', 'sales'),
        ('customer', 'customer_service'), ('support', 'customer_service'),
        ('admin', 'administrative'), ('clerk', 'administrative'),
        ('writer', 'content'), ('content', 'content'),
        ('finance', 'finance'), ('accounting', 'finance'),
        ('graduate', 'entry'), ('entry', 'entry'), ('junior', 'entry'),
    ]
    for keyword, role in role_patterns:
        if keyword in query_lower:
            role_type = role
            break
    
    return {
        'technical_skills': list(tech_found),
        'soft_skills': list(soft_found),
        'role_type': role_type,
        'duration_max_minutes': duration_limit,
        'needs_cognitive_test': needs_cognitive,
        'needs_personality_test': needs_personality or len(soft_found) > 0,
        'key_search_terms': list(tech_found)[:5] + list(soft_found)[:3],
        'enhanced_query': query[:200],
        'source': 'keyword_fallback'
    }


def analyze_query(query: str) -> dict:
    """Analyze query using Groq with keyword fallback"""
    
    # Try Groq first
    groq_result = analyze_with_groq(query)
    
    if groq_result:
        groq_result['source'] = 'groq_llm'
        return groq_result
    
    # Fallback to keywords
    print("  Using keyword fallback analysis")
    return analyze_with_keywords(query)


def get_balanced_recommendations(query: str, max_results: int = 10) -> list:
    """Get balanced recommendations using LLM-enhanced analysis"""
    global store
    
    # Ensure vector store is available (lazy init for WSGI servers)
    if store is None:
        try:
            print("  Vector store is not initialized; attempting to initialize...")
            initialize()
        except Exception as e:
            print(f"  Initialization attempt failed: {e}")
            raise RuntimeError("Server not ready: vector store unavailable. Check server logs or restart the application.")
    
    # Handle URL input
    if is_url(query.strip()):
        print("  Detected URL input, extracting content...")
        extracted = extract_text_from_url(query.strip())
        if extracted:
            query = extracted
            print(f"  Extracted {len(query)} characters from URL")
    
    # Analyze query (Groq or fallback)
    analysis = analyze_query(query)
    
    tech_skills = analysis.get('technical_skills', [])
    soft_skills = analysis.get('soft_skills', [])
    role_type = analysis.get('role_type', 'general')
    duration_limit = analysis.get('duration_max_minutes')
    needs_cognitive = analysis.get('needs_cognitive_test', False)
    needs_personality = analysis.get('needs_personality_test', False)
    key_terms = analysis.get('key_search_terms', [])
    enhanced_query = analysis.get('enhanced_query', query)
    
    print(f"  Role: {role_type}, Tech: {len(tech_skills)}, Soft: {len(soft_skills)}")
    print(f"  Cognitive: {needs_cognitive}, Personality: {needs_personality}")
    print(f"  Source: {analysis.get('source', 'unknown')}")
    
    # Build search queries
    search_queries = []
    
    # Primary: enhanced query from LLM or original
    search_queries.append(enhanced_query[:500])
    
    # Technical skills query
    if tech_skills:
        tech_query = ' '.join(tech_skills[:7]) + ' technical programming test assessment'
        search_queries.append(tech_query)
    
    # Soft skills query
    if soft_skills:
        soft_query = ' '.join(soft_skills[:5]) + ' personality behavioral assessment'
        search_queries.append(soft_query)
    
    # Key terms query
    if key_terms:
        search_queries.append(' '.join(key_terms[:8]) + ' assessment test')
    
    # Cognitive query
    if needs_cognitive:
        search_queries.append('cognitive aptitude reasoning numerical verbal logical ability test')
    
    # Personality query
    if needs_personality:
        search_queries.append('personality behavior traits workplace assessment OPQ')
    
    # Role-specific queries
    role_queries = {
        'technical': 'software developer programming coding technical skills assessment',
        'analyst': 'data analysis analytical numerical reasoning business intelligence',
        'sales': 'sales personality motivation customer relationship assessment',
        'customer_service': 'customer service communication english support assessment',
        'managerial': 'leadership management decision making executive assessment',
        'administrative': 'administrative clerical office skills attention detail',
        'content': 'verbal written english communication content writing',
        'finance': 'numerical financial accounting analytical assessment',
        'entry': 'graduate entry level aptitude personality potential assessment',
    }
    if role_type in role_queries:
        search_queries.append(role_queries[role_type])
    
    # Execute searches and collect results
    all_results = []
    seen_urls = set()
    
    for sq in search_queries:
        results = store.search(sq, n_results=30)
        for r in results:
            norm_url = normalize_url(r['url'])
            if norm_url not in seen_urls:
                all_results.append(r)
                seen_urls.add(norm_url)
    
    print(f"  Total unique results from searches: {len(all_results)}")
    
    # Categorize by type
    k_type = [r for r in all_results if 'K' in r.get('test_type_raw', '')]
    p_type = [r for r in all_results if 'P' in r.get('test_type_raw', '')]
    a_type = [r for r in all_results if 'A' in r.get('test_type_raw', '')]
    b_type = [r for r in all_results if 'B' in r.get('test_type_raw', '')]
    c_type = [r for r in all_results if 'C' in r.get('test_type_raw', '')]
    
    print(f"  By type: K={len(k_type)}, P={len(p_type)}, A={len(a_type)}, B={len(b_type)}, C={len(c_type)}")
    
    final_results = []
    final_urls = set()
    
    # Determine if we need balance
    needs_balance = (len(tech_skills) > 0 and len(soft_skills) > 0) or \
                    (len(tech_skills) > 0 and needs_personality)
    
    # Balanced selection: interleave K and P types
    if needs_balance and k_type and p_type:
        print("  Applying balanced K + P selection...")
        k_sorted = sorted(k_type, key=lambda x: x.get('score', 0), reverse=True)
        p_sorted = sorted(p_type, key=lambda x: x.get('score', 0), reverse=True)
        
        k_idx, p_idx = 0, 0
        while len(final_results) < max_results:
            # Add K type
            while k_idx < len(k_sorted):
                url = normalize_url(k_sorted[k_idx]['url'])
                if url not in final_urls:
                    final_results.append(k_sorted[k_idx])
                    final_urls.add(url)
                    k_idx += 1
                    break
                k_idx += 1
            
            if len(final_results) >= max_results:
                break
            
            # Add P type
            while p_idx < len(p_sorted):
                url = normalize_url(p_sorted[p_idx]['url'])
                if url not in final_urls:
                    final_results.append(p_sorted[p_idx])
                    final_urls.add(url)
                    p_idx += 1
                    break
                p_idx += 1
            
            if k_idx >= len(k_sorted) and p_idx >= len(p_sorted):
                break
    
    # Add cognitive tests if needed
    if needs_cognitive:
        a_sorted = sorted(a_type, key=lambda x: x.get('score', 0), reverse=True)
        for r in a_sorted[:3]:
            url = normalize_url(r['url'])
            if url not in final_urls and len(final_results) < max_results:
                final_results.append(r)
                final_urls.add(url)
    
    # Fill remaining from all results by score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    for r in all_results:
        if len(final_results) >= max_results:
            break
        url = normalize_url(r['url'])
        if url not in final_urls:
            final_results.append(r)
            final_urls.add(url)
    
    # Apply duration filter if specified
    if duration_limit:
        filtered = [r for r in final_results if r.get('duration', 30) <= duration_limit]
        if len(filtered) >= 5:
            final_results = filtered[:max_results]
            print(f"  Duration filter: {len(final_results)} assessments under {duration_limit} min")
        else:
            final_results = final_results[:max_results]
    else:
        final_results = final_results[:max_results]
    
    # Log final distribution
    final_k = len([r for r in final_results if 'K' in r.get('test_type_raw', '')])
    final_p = len([r for r in final_results if 'P' in r.get('test_type_raw', '')])
    final_a = len([r for r in final_results if 'A' in r.get('test_type_raw', '')])
    print(f"  Final: {len(final_results)} results (K={final_k}, P={final_p}, A={final_a})")
    
    # Format for API response
    formatted = []
    for r in final_results:
        test_types = r.get('test_type', ['General'])
        if isinstance(test_types, str):
            test_types = [test_types]
        
        formatted.append({
            "name": r.get('name', 'Unknown'),
            "url": r.get('url', '#'),
            "description": r.get('description', f"Assessment for evaluating {r.get('name', 'skills')}"),
            "test_type": test_types,
            "duration": r.get('duration', 30),
            "remote_support": "Yes" if r.get('remote_testing') else "No",
            "adaptive_support": "Yes" if r.get('adaptive_irt') else "No"
        })
    
    return formatted


# =============================================================================
# Startup hook
# =============================================================================

@app.before_first_request
def startup():
    """Ensure initialization when running under WSGI (gunicorn, uWSGI, etc.)"""
    global store, groq_client
    if store is None:
        try:
            print("Startup: initializing system...")
            initialize()
        except Exception as e:
            # Initialization may fail in some environments; keep server up but provide clear logs
            print(f"Startup initialization failed: {e}")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return render_template('index.html', query="", recommendations=None)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Recommendation endpoint - returns JSON for API, HTML for browser"""
    
    is_api = (
        request.method == 'POST' or
        request.headers.get('Accept', '').startswith('application/json') or
        request.args.get('format') == 'json'
    )
    
    if request.method == 'POST':
        data = request.get_json(silent=True)
        query = data.get('query', '') if data else ''
    else:
        query = request.args.get('query', '')
    
    if not query or not query.strip():
        if is_api:
            return jsonify({"error": "Query is required"}), 400
        return render_template('index.html',
                             error_message="Please enter a job description, query, or URL.",
                             query="", recommendations=None)
    
    print(f"\n{'='*60}")
    print(f"Query: {query[:100]}...")
    print('='*60)
    
    try:
        results = get_balanced_recommendations(query, max_results=10)
        print(f"  Returning {len(results)} recommendations\n")
        
        if is_api:
            return jsonify({"recommended_assessments": results})
        
        return render_template('index.html',
                             recommendations=results,
                             query=query,
                             error_message=None)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        if is_api:
            return jsonify({"error": str(e)}), 500
        
        return render_template('index.html',
                             error_message=f"An error occurred: {str(e)}",
                             query=query,
                             recommendations=None)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    initialize()
    
    # Get port from environment (Hugging Face uses 7860)
    port = int(os.environ.get('PORT', 7860))
    
    print("\n" + "="*60)
    print("SHL Assessment Recommendation API")
    print("="*60)
    print(f"  Web UI:     http://127.0.0.1:{port}/")
    print(f"  Health:     http://127.0.0.1:{port}/health")
    print(f"  API (GET):  http://127.0.0.1:{port}/recommend?query=...&format=json")
    print(f"  API (POST): POST /recommend with {{\"query\": \"...\"}}")
    print("="*60 + "\n")
    
    # In production, gunicorn handles this. Debug only for local dev.
    app.run(debug=False, host='0.0.0.0', port=port)