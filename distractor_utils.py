import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from typing import List, Dict, Optional

## Build distracting context for a PubMedQA question using context and labels from other questions

def load_pubmedqa_dataset(subset: str = "pqa_labeled", split: str = "train") -> List[Dict]:
    """
    Load PubMedQA dataset from HuggingFace.
    
    Args:
        subset: Dataset subset (default: "pqa_labeled")
        split: Dataset split (default: "train")
    
    Returns:
        List of formatted items with context and label
    """
    dataset = load_dataset("qiaojin/PubMedQA", subset, split=split)
    return format_pubmedqa_for_distractors(dataset)

def format_pubmedqa_for_distractors(dataset) -> List[Dict]:
    """
    Format PubMedQA dataset items for use with build_distractors.
    
    Args:
        dataset: HuggingFace dataset object
    
    Returns:
        List of dicts with 'context', 'label', and 'qid' keys
    """
    formatted_items = []
    
    for i, item in enumerate(dataset):
        # Extract contexts from the nested structure
        contexts = item.get('context', {}).get('contexts', [])
        # Join all context strings into a single text
        context_text = ' '.join(contexts) if contexts else ""
        
        # Get the final decision (yes/no/maybe)
        label = item.get('final_decision', '').lower().strip()
        
        formatted_item = {
            'context': context_text,
            'label': label,
            'qid': item.get('pubid', f'item_{i}')  # Use pubid if available, otherwise generate ID
        }
        
        formatted_items.append(formatted_item)
    
    return formatted_items

def create_distractors_for_pubmedqa_item(qid: str, subset: str = "pqa_labeled", 
                                        split: str = "train", n_benign: int = 2, 
                                        n_nearmiss: int = 2, rng: int = 0) -> List[Dict]:
    """
    Create distractors for a specific PubMedQA item by ID.
    
    Args:
        qid: Question ID to find distractors for
        subset: Dataset subset (default: "pqa_labeled")
        split: Dataset split (default: "train")
        n_benign: Number of benign distractors (same label)
        n_nearmiss: Number of near-miss distractors (different label)
        rng: Random seed
    
    Returns:
        List of distractor items with gold item and distractors
    """
    items = load_pubmedqa_dataset(subset=subset, split=split)
    
    # Find the gold item by qid
    gold_idx = None
    for i, item in enumerate(items):
        if item['qid'] == qid:
            gold_idx = i
            break
    
    if gold_idx is None:
        raise ValueError(f"Question ID '{qid}' not found in dataset")
    
    return build_distractors(items, gold_idx, n_benign=n_benign, n_nearmiss=n_nearmiss, rng=rng)

def build_distractors(items, gold_idx, label_key="label", text_key="context",
                      n_benign=2, n_nearmiss=2, rng=0):
    """
    Build distractors for PubMedQA items with yes/no/maybe labels.
    
    Args:
        items: list of dicts, each with 'context' and 'label' and maybe 'qid'
        gold_idx: index of the current item in items
        label_key: key for label field (default: "label")
        text_key: key for text field (default: "context")
        n_benign: number of benign distractors (same label)
        n_nearmiss: number of near-miss distractors (different label)
        rng: random seed
    
    Returns:
        List of distractor items with gold item and distractors
    """
    rng = np.random.RandomState(rng)
    
    # Filter out items with empty contexts
    valid_items = [(i, item) for i, item in enumerate(items) if item[text_key].strip()]
    if not valid_items:
        raise ValueError("No valid items with non-empty contexts found")
    
    # Create mapping from original indices to valid indices
    orig_to_valid = {orig_idx: valid_idx for valid_idx, (orig_idx, _) in enumerate(valid_items)}
    valid_gold_idx = orig_to_valid.get(gold_idx)
    
    if valid_gold_idx is None:
        raise ValueError(f"Gold item at index {gold_idx} has empty context")
    
    valid_texts = [item[text_key] for _, item in valid_items]
    
    # Use more lenient TF-IDF parameters for better similarity computation
    tfidf = TfidfVectorizer(min_df=1, max_df=0.95, stop_words='english').fit(valid_texts)
    X = tfidf.transform(valid_texts)

    gold = valid_items[valid_gold_idx][1]
    gold_vec = X[valid_gold_idx]
    sims = cosine_similarity(gold_vec, X).ravel()
    order = np.argsort(-sims)

    benign, nearmiss = [], []
    pubmedqa_labels = {'yes', 'no', 'maybe'}
    
    for j in order:
        if j == valid_gold_idx: 
            continue
            
        candidate_item = valid_items[j][1]
        candidate_label = candidate_item[label_key].lower().strip()
        gold_label = gold[label_key].lower().strip()
        
        # Skip items with invalid labels
        if candidate_label not in pubmedqa_labels or gold_label not in pubmedqa_labels:
            continue
            
        same_label = (candidate_label == gold_label)
        similarity = sims[j]
        
        # Adjusted thresholds for PubMedQA context similarity
        if not same_label and similarity > 0.1 and len(nearmiss) < n_nearmiss:
            nearmiss.append({
                "text": candidate_item[text_key], 
                "kind": "near_miss", 
                "sim": float(similarity),
                "label": candidate_label,
                "qid": candidate_item.get('qid', 'unknown')
            })
        elif same_label and similarity > 0.05 and len(benign) < n_benign:
            benign.append({
                "text": candidate_item[text_key], 
                "kind": "benign", 
                "sim": float(similarity),
                "label": candidate_label,
                "qid": candidate_item.get('qid', 'unknown')
            })
            
        if len(benign) == n_benign and len(nearmiss) == n_nearmiss:
            break

    # Create the bundle with gold item first
    bundle = [{
        "text": gold[text_key], 
        "kind": "gold", 
        "sim": 1.0,
        "label": gold[label_key],
        "qid": gold.get('qid', 'unknown')
    }] + nearmiss + benign
    
    # Shuffle only the distractors, keep gold first
    distractors = bundle[1:]
    rng.shuffle(distractors)
    
    return [bundle[0]] + distractors

def example_usage():
    """
    Example usage of the PubMedQA distractor functions.
    """
    print("Loading PubMedQA dataset...")
    items = load_pubmedqa_dataset(subset="pqa_labeled", split="train")
    print(f"Loaded {len(items)} items")
    
    if len(items) > 0:
        # Example 1: Create distractors for the first item
        print(f"\nExample 1: Creating distractors for item 0")
        print(f"Gold item label: {items[0]['label']}")
        print(f"Gold item context (first 100 chars): {items[0]['context'][:100]}...")
        
        distractors = build_distractors(items, 0, n_benign=2, n_nearmiss=2, rng=42)
        
        print(f"\nFound {len(distractors)} items (1 gold + distractors):")
        for i, item in enumerate(distractors):
            print(f"  {i+1}. {item['kind']} (label: {item['label']}, sim: {item['sim']:.3f})")
            print(f"     Context: {item['text'][:80]}...")
        
        # Example 2: Create distractors by QID if available
        if 'qid' in items[0] and items[0]['qid'] != 'item_0':
            print(f"\nExample 2: Creating distractors by QID: {items[0]['qid']}")
            try:
                distractors_by_qid = create_distractors_for_pubmedqa_item(
                    items[0]['qid'], n_benign=1, n_nearmiss=1, rng=42
                )
                print(f"Successfully created distractors by QID")
            except Exception as e:
                print(f"Error creating distractors by QID: {e}")

if __name__ == "__main__":
    example_usage()
