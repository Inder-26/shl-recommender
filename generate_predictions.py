"""
Generate Predictions CSV for SHL Assessment Recommender
Output format: Query, Assessment_url
"""

import pandas as pd
import os
import sys

# Add parent directory for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import initialize, get_balanced_recommendations


# Configuration
TEST_FILE = "Gen_AI Dataset_Test_Set.xlsx"
SHEET_NAME = "Test-Set"
OUTPUT_FILE = "evaluation/predictions_test_set.csv"
NUM_RECOMMENDATIONS = 10


def main():
    """Generate predictions CSV"""
    
    print("="*60)
    print("GENERATING PREDICTIONS CSV")
    print("="*60)
    
    # Initialize (includes Groq)
    print("\n[1] Initializing system...")
    initialize()
    
    # Load test queries
    print(f"\n[2] Loading queries from {TEST_FILE}...")
    
    if not os.path.exists(TEST_FILE):
        print(f"  Error: File not found: {TEST_FILE}")
        return
    
    df = pd.read_excel(TEST_FILE, sheet_name=SHEET_NAME)
    
    # Find query column
    query_col = None
    for col in df.columns:
        if 'query' in col.lower():
            query_col = col
            break
    
    if not query_col:
        print(f"  Error: Could not find Query column. Found: {df.columns.tolist()}")
        return
    
    queries = df[query_col].tolist()
    print(f"  Loaded {len(queries)} queries")
    
    # Generate recommendations
    print(f"\n[3] Generating recommendations...")
    print("-"*60)
    
    rows = []
    
    for i, query in enumerate(queries, 1):
        query = str(query).strip()
        
        if not query or query.lower() == 'nan':
            print(f"  [{i}/{len(queries)}] Skipped (empty)")
            continue
        
        print(f"\n  [{i}/{len(queries)}] Processing...")
        
        # Get recommendations
        results = get_balanced_recommendations(query, max_results=NUM_RECOMMENDATIONS)
        
        # Add each as a row
        for r in results:
            rows.append({
                "Query": query,
                "Assessment_url": r['url']
            })
        
        print(f"  Added {len(results)} recommendations")
    
    # Save CSV
    print(f"\n[4] Saving to {OUTPUT_FILE}...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df_out = pd.DataFrame(rows)
    df_out = df_out[['Query', 'Assessment_url']]
    df_out.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df_out)}")
    print(f"  Unique queries: {df_out['Query'].nunique()}")
    print(f"  Recommendations per query: {len(df_out) / max(df_out['Query'].nunique(), 1):.1f}")
    print("="*60)


if __name__ == '__main__':
    main()