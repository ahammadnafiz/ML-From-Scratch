import json
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Initialize NLP and model
nlp = spacy.load("en_core_web_sm")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess_text(file_path):
    """Split text into sentences using spaCy"""
    with open(file_path, "r") as f:
        text = f.read()
    return [sent.text for sent in nlp(text).sents]

def extract_embeddings(sentences):
    """Case-sensitive embedding extraction with case-insensitive index"""
    word_embeddings = defaultdict(list)
    case_mapping = defaultdict(set)
    
    for sent in sentences:
        spacy_doc = nlp(sent)
        spacy_tokens = [
            {"text": token.text, "start": token.idx, "end": token.idx + len(token.text)}
            for token in spacy_doc 
            if not token.is_punct and not token.is_stop and len(token.text) > 1
        ]
        
        inputs = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offset_mapping = inputs.pop('offset_mapping').squeeze(0).tolist()
        
        valid_indices = [i for i, (s, e) in enumerate(offset_mapping) if s != e]
        offset_mapping = [offset_mapping[i] for i in valid_indices]
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.squeeze(0)[valid_indices]
        
        for token in spacy_tokens:
            original_text = token["text"]
            lower_text = original_text.lower()
            
            matched_indices = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if start >= token["start"] and start < token["end"]
            ]
            
            if matched_indices:
                token_embeddings = embeddings[matched_indices].numpy()
                word_embeddings[lower_text].append(np.mean(token_embeddings, axis=0))
                case_mapping[lower_text].add(original_text)
    
    # Preserve original casing in final output
    final_embeddings = {}
    for lower_word, embbs in word_embeddings.items():
        final_embeddings[lower_word] = {
            "embeddings": np.mean(embbs, axis=0),
            "variants": list(case_mapping[lower_word])
        }
    return final_embeddings

def build_co_occurrence(sentences, window_size=3):
    """Case-sensitive co-occurrence tracking with normalization"""
    co_occur = defaultdict(lambda: defaultdict(int))
    case_mapping = defaultdict(set)
    
    for sent in sentences:
        words = [
            token.text 
            for token in nlp(sent) 
            if not token.is_punct and not token.is_stop and len(token.text) > 1
        ]
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            case_mapping[lower_word].add(word)
            
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j and word != words[j]:
                    co_occur[lower_word][words[j].lower()] += 1
    
    return co_occur, case_mapping

def create_knowledge_base(embeddings, co_occur, case_mapping, similarity_threshold=0.7):
    """Knowledge base with preserved casing and case-insensitive lookup"""
    knowledge_base = defaultdict(lambda: defaultdict(list))
    words = list(embeddings.keys())
    
    # Semantic similarities
    if words:
        sim_matrix = cosine_similarity(np.array([embeddings[w]["embeddings"] for w in words]))
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j and sim_matrix[i][j] > similarity_threshold:
                    for variant in embeddings[w1]["variants"]:
                        knowledge_base[variant][w2].append({
                            "type": "semantic",
                            "score": float(sim_matrix[i][j]),
                            "original": embeddings[w2]["variants"]
                        })
    
    # Syntactic co-occurrences
    for w1_lower, neighbors in co_occur.items():
        for variant in case_mapping.get(w1_lower, [w1_lower]):
            for w2_lower, count in neighbors.items():
                knowledge_base[variant][w2_lower].append({
                    "type": "syntactic",
                    "score": float(count / (sum(neighbors.values()) or 1)),
                    "original": list(case_mapping.get(w2_lower, {w2_lower}))
                })
    
    return knowledge_base

def visualize_word_graph(knowledge_base, target_word):
    """Visualization with original casing"""
    # Case-insensitive lookup
    target_lower = target_word.lower()
    matches = [word for word in knowledge_base.keys() if word.lower() == target_lower]
    
    if not matches:
        print(f"Word '{target_word}' not found in knowledge base!")
        return
    
    # Use first matched casing
    canonical_word = matches[0]
    
    G = nx.MultiDiGraph()
    for neighbor, relations in knowledge_base[canonical_word].items():
        for rel in relations:
            G.add_edge(canonical_word, neighbor, 
                      weight=rel["score"]*10,
                      type=rel["type"],
                      original=rel.get("original", [neighbor]))
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_colors = ["red" if data["type"] == "semantic" else "blue" 
                   for _, _, data in G.edges(data=True)]
    
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True,
            edge_color=edge_colors,
            node_color="lightgreen",
            node_size=3000,
            font_size=14,
            arrowsize=25,
            width=[d['weight'] for _, _, d in G.edges(data=True)])
    
    plt.legend(handles=[
        plt.Line2D([0], [0], color="red", lw=4, label="Semantic Similarity"),
        plt.Line2D([0], [0], color="blue", lw=4, label="Syntactic Co-occurrence")
    ], loc='best')
    
    plt.title(f"Semantic Network for '{canonical_word}'", fontsize=16)
    plt.show()

def default_to_regular(d):
    """Recursively convert defaultdicts and numpy types for JSON"""
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, list):
        return [default_to_regular(x) for x in d]
    elif isinstance(d, dict):
        return {k: default_to_regular(v) for k, v in d.items()}
    return d

if __name__ == "__main__":
    target_word = input("Enter a word to analyze: ").strip()
    if not target_word:
        print("Please enter a valid word")
        exit()
    
    sentences = preprocess_text("test.txt")
    
    embeddings = extract_embeddings(sentences)
    co_occur, case_mapping = build_co_occurrence(sentences)
    knowledge_base = create_knowledge_base(embeddings, co_occur, case_mapping)
    
    visualize_word_graph(knowledge_base, target_word)
    
    # Save target word's knowledge base entries to JSON
    target_lower = target_word.lower()
    matches = [word for word in knowledge_base if word.lower() == target_lower]
    
    if not matches:
        print(f"Target word '{target_word}' not found in knowledge base. JSON file not created.")
    else:
        # Collect all entries matching the target word (case-insensitive)
        target_data = {word: knowledge_base[word] for word in matches}
        # Convert defaultdicts to regular dicts
        target_data = default_to_regular(target_data)
        
        filename = f"{target_lower}_knowledge_base.json"
    
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved knowledge base entries for '{target_word}' to {filename}")