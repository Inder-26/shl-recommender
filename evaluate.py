"""
Evaluation Script for SHL Assessment Recommender
Calculates Mean Recall@K on the training set
"""

import pandas as pd
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import initialize, get_balanced_recommendations


def normalize_url(url: str) -> str:
    """Normalize URL for comparison"""
    if not url:
        return ""
    url = str(url).strip().lower().rstrip('/')
    if '/products/product-catalog/' in url and '/solutions/products/' not in url:
        url = url.replace('/products/product-catalog/', '/solutions/products/product-catalog/')
    return url


def calculate_recall_at_k(predicted_urls: list, relevant_urls: set, k: int = 10):
    """Calculate Recall@K with normalized URLs"""
    if not relevant_urls:
        return 0.0, 0
    
    predicted_normalized = set(normalize_url(u) for u in predicted_urls[:k])
    relevant_normalized = set(normalize_url(u) for u in relevant_urls)
    
    hits = len(predicted_normalized.intersection(relevant_normalized))
    recall = hits / len(relevant_normalized)
    
    return recall, hits


def main():
    """Run evaluation on training set"""
    
    print("="*60)
    print("EVALUATION: Mean Recall@K on Training Set")
    print("="*60)
    
    # Configuration
    train_file = "Gen_AI Dataset_Train_Set.xlsx"
    sheet_name = "Train-Set"
    k = 10
    
    # Initialize app (includes Groq)
    print("\n[1] Initializing system...")
    initialize()
    
    # Load training data
    print(f"\n[2] Loading training data from {train_file}...")
    
    if not os.path.exists(train_file):
        print(f"  Error: File not found: {train_file}")
        return
    
    df = pd.read_excel(train_file, sheet_name=sheet_name)
    print(f"  Loaded {len(df)} rows")
    
    # Find columns
    query_col = None
    url_col = None
    for col in df.columns:
        if 'query' in col.lower():
            query_col = col
        if 'url' in col.lower() or 'assessment' in col.lower():
            url_col = col
    
    if not query_col or not url_col:
        print(f"  Error: Could not find columns. Found: {df.columns.tolist()}")
        return
    
    # Group by query
    query_to_urls = {}
    for _, row in df.iterrows():
        query = str(row[query_col]).strip()
        url = str(row[url_col]).strip()
        if query and url and query.lower() != 'nan':
            if query not in query_to_urls:
                query_to_urls[query] = set()
            query_to_urls[query].add(url)
    
    print(f"  Found {len(query_to_urls)} unique queries")
    
    # Evaluate
    print(f"\n[3] Evaluating (k={k})...")
    print("-"*60)
    
    recalls = []
    total_hits = 0
    total_relevant = 0
    
    for i, (query, relevant_urls) in enumerate(query_to_urls.items(), 1):
        print(f"\n  Query {i}/{len(query_to_urls)}: {query[:50]}...")
        print(f"  Relevant URLs: {len(relevant_urls)}")
        
        # Get predictions using the full pipeline
        results = get_balanced_recommendations(query, max_results=k)
        predicted_urls = [r['url'] for r in results]
        
        # Calculate recall
        recall, hits = calculate_recall_at_k(predicted_urls, relevant_urls, k)
        recalls.append(recall)
        total_hits += hits
        total_relevant += len(relevant_urls)
        
        print(f"  Predicted: {len(predicted_urls)} | Hits: {hits} | Recall@{k}: {recall:.4f}")
        
        # Show hits
        if hits > 0:
            pred_norm = set(normalize_url(u) for u in predicted_urls[:k])
            rel_norm = set(normalize_url(u) for u in relevant_urls)
            matching = pred_norm.intersection(rel_norm)
            for m in list(matching)[:2]:
                print(f"    HIT: ...{m[-50:]}")
    
    # Results
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Number of queries: {len(recalls)}")
    print(f"  K value: {k}")
    print(f"  Total hits: {total_hits} / {total_relevant}")
    print(f"  Individual recalls: {[f'{r:.3f}' for r in recalls]}")
    print(f"\n  >>> Mean Recall@{k}: {mean_recall:.4f} ({mean_recall*100:.2f}%) <<<")
    print("="*60)
    
    return mean_recall


if __name__ == '__main__':
    main()