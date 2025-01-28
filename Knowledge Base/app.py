import streamlit as st
import json
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

# # Custom CSS for modern layout
st.markdown("""
    <style>
        .main {
            max-width: 2500px;
            padding: 2rem;
        }
        .header {
            background: linear-gradient(45deg, #6366f1, #8b5cf6);
            padding: 3rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: #6366f1;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# Modern header
st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5rem; margin: 0;">TEXT RELATION EXPLORER</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Semantic & Syntactic Relationship Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Initialize NLP and model with improved configuration
@st.cache_resource
def load_nlp():
    nlp = spacy.load("en_core_web_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

@st.cache_resource
def load_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model

nlp = load_nlp()
tokenizer, model = load_model()

# Enhanced text processing with lemmatization
def preprocess_text(text_content):
    doc = nlp(text_content)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

# Improved embedding extraction with layer aggregation
def extract_embeddings(sentences):
    word_embeddings = defaultdict(list)
    case_mapping = defaultdict(set)
    
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offset_mapping = inputs.pop('offset_mapping').squeeze(0).tolist()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[-4:]
        embeddings = torch.mean(torch.stack(hidden_states), dim=0).squeeze(0)
        
        spacy_doc = nlp(sent)
        for token in spacy_doc:
            if token.is_punct or token.is_stop or len(token.text) < 2:
                continue
                
            token_start = token.idx
            token_end = token.idx + len(token.text)
            token_vectors = []
            
            for idx, (start, end) in enumerate(offset_mapping):
                if end <= token_start:
                    continue
                if start >= token_end:
                    break
                token_vectors.append(embeddings[idx].numpy())
            
            if token_vectors:
                lower_text = token.text.lower()
                avg_vector = np.mean(token_vectors, axis=0)
                word_embeddings[lower_text].append(avg_vector)
                case_mapping[lower_text].add(token.text)
    
    final_embeddings = {}
    for word, vectors in word_embeddings.items():
        final_embeddings[word] = {
            "embeddings": np.mean(vectors, axis=0),
            "variants": list(case_mapping[word])
        }
    return final_embeddings

# PMI-enhanced co-occurrence analysis
def build_co_occurrence(sentences, window_size=3):
    co_occur = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)
    total_pairs = 0
    
    for sent in sentences:
        words = [token.text.lower() for token in nlp(sent) 
                if not token.is_punct and not token.is_stop]
        
        for i, word in enumerate(words):
            word_counts[word] += 1
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j and word != words[j]:
                    co_occur[word][words[j]] += 1
                    total_pairs += 1

    pmi_matrix = defaultdict(lambda: defaultdict(float))
    for w1, neighbors in co_occur.items():
        for w2, count in neighbors.items():
            p_xy = count / total_pairs
            p_x = word_counts[w1] / total_pairs
            p_y = word_counts[w2] / total_pairs
            pmi = np.log(p_xy / (p_x * p_y + 1e-8))
            pmi_matrix[w1][w2] = max(pmi, 0)
    
    return pmi_matrix, word_counts

# Enhanced knowledge base with relation weighting
def create_knowledge_base(embeddings, co_occur, similarity_threshold=0.7):
    knowledge_base = defaultdict(lambda: defaultdict(list))
    words = list(embeddings.keys())
    
    if words:
        embeddings_matrix = np.array([embeddings[w]["embeddings"] for w in words])
        sim_matrix = cosine_similarity(embeddings_matrix)
        
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j and sim_matrix[i][j] > similarity_threshold:
                    for variant in embeddings[w1]["variants"]:
                        knowledge_base[variant][w2].append({
                            "type": "semantic",
                            "score": float(sim_matrix[i][j]),
                            "weight": 0.7
                        })
    
    for w1, neighbors in co_occur.items():
        if w1 in embeddings:
            for variant in embeddings[w1]["variants"]:
                for w2, pmi_score in neighbors.items():
                    knowledge_base[variant][w2].append({
                        "type": "syntactic",
                        "score": float(pmi_score),
                        "weight": 0.3
                    })
    
    return knowledge_base

# Modern visualization with improved styling
def visualize_with_pyvis(knowledge_base, target_word):
    target_lower = target_word.lower()
    matches = [word for word in knowledge_base.keys() if word.lower() == target_lower]
    
    if not matches:
        return None
    
    G = nx.MultiDiGraph()
    edge_colors = {
        "semantic": "#6366f1",
        "syntactic": "#10b981"
    }
    
    for neighbor, relations in knowledge_base[matches[0]].items():
        for rel in relations:
            G.add_edge(
                matches[0], neighbor,
                title=f"{rel['type'].title()} ({rel['score']:.2f})",
                color=edge_colors[rel["type"]],
                weight=rel["score"] * 0.1,
                type=rel["type"]
            )
    
    net = Network(height="800px", width="100%", notebook=True, cdn_resources="remote")
    net.from_nx(G)
    
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 18},
            "shape": "dot",
            "size": 25
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "scaling": {"min": 1, "max": 5}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "springLength": 100,
                "springConstant": 0.01
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }
    }
    """)
    
    return net

# Modern Streamlit layout
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload Text File", type=["txt"])
    with col2:
        target_word = st.text_input("🎯 Target Word", key='target_word')

process_btn = st.button("Analyze Text →", type="primary")

# Enhanced sidebar with expanders
with st.sidebar:
    st.markdown("## Configuration Panel")
    with st.expander("🧠 Semantic Settings", expanded=True):
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.0, 1.0, 0.7, 0.05
        )
    
    with st.expander("🔗 Syntactic Settings"):
        window_size = st.slider(
            "Context Window Size",
            1, 5, 3
        )
    
    st.markdown("---")
    st.markdown("### Application Info")
    st.info("""
    This application analyzes text using:
    - BERT embeddings for semantic similarity
    - PMI-enhanced co-occurrence analysis
    - Interactive network visualizations
    """)

# Processing pipeline
if uploaded_file and process_btn:
    current_content = uploaded_file.read().decode()
    uploaded_file.seek(0)

    if 'file_content' not in st.session_state or st.session_state.file_content != current_content:
        with st.spinner("🔍 Analyzing text relationships..."):
            sentences = preprocess_text(current_content)
            embeddings = extract_embeddings(sentences)
            co_occur, _ = build_co_occurrence(sentences, window_size)
            knowledge_base = create_knowledge_base(embeddings, co_occur, similarity_threshold)
            
            st.session_state.knowledge_base = knowledge_base
            st.session_state.file_content = current_content

# Results visualization
if 'knowledge_base' in st.session_state and st.session_state.target_word.strip():
    st.markdown("---")
    st.markdown("## Visualization Results")
    
    with st.container():
        net = visualize_with_pyvis(st.session_state.knowledge_base, st.session_state.target_word)
        if net:
            html_file = "graph.html"
            net.save_graph(html_file)
            st.components.v1.html(open(html_file).read(), height=800)
        else:
            st.error("Target word not found in analysis results")

    # Enhanced export section
    st.markdown("---")
    with st.expander("📤 Export Results", expanded=True):
        st.markdown("### Download Analysis Data")
        if st.button("Generate JSON Report"):
            target_lower = st.session_state.target_word.lower()
            matches = [w for w in st.session_state.knowledge_base if w.lower() == target_lower]
            
            if matches:
                export_data = {
                    "target": matches[0],
                    "relations": st.session_state.knowledge_base[matches[0]]
                }
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    "⬇️ Download JSON",
                    json_str,
                    file_name=f"{target_lower}_relations.json",
                    mime="application/json"
                )