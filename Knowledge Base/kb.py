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
    """Extract accurate word embeddings using BERT-spaCy alignment"""
    word_embeddings = defaultdict(list)
    
    for sent in sentences:
        # Get spaCy tokens with offsets
        spacy_doc = nlp(sent)
        spacy_tokens = [
            {"text": token.text, "start": token.idx, "end": token.idx + len(token.text)}
            for token in spacy_doc if not token.is_punct and not token.is_stop
        ]
        
        # Get BERT tokenization with offsets
        inputs = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offset_mapping = inputs.pop('offset_mapping').squeeze(0).tolist()  # Remove offset mapping from inputs
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.squeeze(0)
        
        # Align spaCy tokens with BERT subwords
        for token in spacy_tokens:
            subword_indices = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if start >= token["start"] and end <= token["end"]
            ]
            
            if subword_indices:
                token_embedding = np.mean([embeddings[idx].numpy() for idx in subword_indices], axis=0)
                word_embeddings[token["text"].lower()].append(token_embedding)  # Lowercase for consistency
    
    return {word: np.mean(embbs, axis=0) for word, embbs in word_embeddings.items()}

def build_co_occurrence(sentences, window_size=3):
    """Calculate co-occurrence between meaningful words"""
    co_occur = defaultdict(lambda: defaultdict(int))
    
    for sent in sentences:
        words = [token.text for token in nlp(sent) if not token.is_punct and not token.is_stop]
        for i, word in enumerate(words):
            for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                if i != j and word != words[j]:
                    co_occur[word][words[j]] += 1
                    
    return co_occur

def create_knowledge_base(embeddings, co_occur, similarity_threshold=0.7):
    """Create enhanced knowledge base with multi-type relationships"""
    knowledge_base = defaultdict(lambda: defaultdict(list))
    words = list(embeddings.keys())
    
    # Semantic similarities
    if words:
        sim_matrix = cosine_similarity(np.array([embeddings[w] for w in words]))
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j and sim_matrix[i][j] > similarity_threshold:
                    knowledge_base[w1][w2].append({
                        "type": "semantic",
                        "score": float(sim_matrix[i][j])
                    })
    
    # Syntactic co-occurrences
    for w1, neighbors in co_occur.items():
        for w2, count in neighbors.items():
            if w2 in embeddings:
                knowledge_base[w1][w2].append({
                    "type": "syntactic",
                    "score": float(count / (sum(neighbors.values()) or 1))
                })
    
    return knowledge_base

def visualize_word_graph(knowledge_base, target_word):
    """Visualize multi-relationship graph with legend"""
    if target_word not in knowledge_base:
        print(f"Word '{target_word}' not found in knowledge base!")
        return
    
    G = nx.MultiDiGraph()
    for neighbor, relations in knowledge_base[target_word].items():
        for rel in relations:
            G.add_edge(target_word, neighbor, 
                      weight=rel["score"], 
                      type=rel["type"])
    
    pos = nx.spring_layout(G)
    edge_colors = ["red" if data["type"] == "semantic" else "blue" 
                   for _, _, data in G.edges(data=True)]
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True,
            edge_color=edge_colors,
            node_color="lightblue",
            node_size=2500,
            font_size=12,
            arrowsize=20)
    
    # Add legend
    plt.legend(handles=[
        plt.Line2D([0], [0], color="red", label="Semantic Similarity"),
        plt.Line2D([0], [0], color="blue", label="Syntactic Co-occurrence")
    ])
    plt.title(f"Semantic Network for '{target_word}'")
    plt.show()

if __name__ == "__main__":
    target_word = input("Enter a word to analyze: ").strip().lower()
    sentences = preprocess_text("test.txt")
    
    embeddings = extract_embeddings(sentences)
    co_occur = build_co_occurrence(sentences)
    knowledge_base = create_knowledge_base(embeddings, co_occur)
    
    # Save results
    output = {
        target_word: {
            neighbor: relations 
            for neighbor, relations in knowledge_base.get(target_word, {}).items()
        }
    }
    
    with open(f"knowledge_{target_word}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    visualize_word_graph(knowledge_base, target_word)