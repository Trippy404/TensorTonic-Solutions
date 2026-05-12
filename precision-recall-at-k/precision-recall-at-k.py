def precision_recall_at_k(recommended, relevant, k):
    # Step 1: Take top-k
    top_k = recommended[:k]
    
    # Step 2: Convert relevant to set for fast lookup
    relevant_set = set(relevant)
    
    # Step 3: Count hits
    hits = sum(1 for item in top_k if item in relevant_set)
    
    # Step 4: Compute precision and recall
    precision = hits / k
    recall = hits / len(relevant)
    
    return [precision, recall]
    
    