import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch  # <-- 1. Import torch

# --- 1. CONFIGURATION & LOADING ---

st.set_page_config(page_title="Trilingual Idiom Finder", page_icon="🗣️")

# Use Streamlit's caching to load the model and data only once
@st.cache_resource
def load_model():
    """Loads a multilingual sentence-transformer model."""
    # This model is good for 50+ languages, including Eng, Hin, and Mar
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data(filepath="idioms.json"):
    """Loads the idiom data from the JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: {filepath} not found. Make sure the file is in the same directory.")
        return []

@st.cache_data
def create_embeddings(_model, data):
    """
    Creates a searchable "corpus" of embeddings.
    We will create one embedding for each phrase (Eng, Hin, Mar)
    and map it back to the original full idiom object.
    """
    corpus_phrases = []
    corpus_mapping = []

    for item in data:
        # Add English phrase
        corpus_phrases.append(item['english_phrase'])
        corpus_mapping.append(item)
        
        # Add Hindi phrase
        corpus_phrases.append(item['hindi_phrase'])
        corpus_mapping.append(item)
        
        # Add Marathi phrase
        corpus_phrases.append(item['marathi_phrase'])
        corpus_mapping.append(item)
        
    # Generate embeddings for all phrases at once (this is fast)
    st.info("Creating search index... this may take a moment on first run.")
    corpus_embeddings = _model.encode(corpus_phrases, convert_to_tensor=True, show_progress_bar=True)
    st.success("Search index ready!")
    
    return corpus_embeddings, corpus_mapping

# --- 2. SEARCH FUNCTION (Corrected) ---

def search_idiom(query, model, embeddings, mapping, top_k=1, min_score=0.6):
    """Performs semantic search."""
    # Encode the user's query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Use cosine-similarity to find the best matches
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # --- START OF FIX ---
    # util.top_k is deprecated. We use torch.topk instead.
    top_results = torch.topk(cos_scores, k=top_k)
    
    # Unpack the best result
    best_score = top_results.values[0].item()
    best_index = top_results.indices[0].item()
    # --- END OF FIX ---
    
    if best_score >= min_score:
        # Return the full idiom object from our mapping
        return mapping[best_index], best_score
    else:
        # No good match found
        return None, None

# --- 3. STREAMLIT UI ---

st.title("🗣️ Trilingual Idiom & Proverb Finder")
st.markdown("Search for an idiom in **English**, **Hindi**, or **Marathi** to find its meaning and equivalents in all three languages.")

# Load all necessary assets
model = load_model()
data = load_data()

if data: # Only proceed if data was loaded successfully
    corpus_embeddings, corpus_mapping = create_embeddings(model, data)

    # User input
    user_query = st.text_input("Enter an idiom or proverb:", placeholder="e.g., to spill the beans or नाच न जाने...")

    if st.button("Search") or user_query:
        if user_query:
            # Perform the search
            result_item, score = search_idiom(user_query, model, corpus_embeddings, corpus_mapping)
            
            if result_item:
                st.success(f"Found a match with {score*100:.0f}% confidence:")
                
                # Display the results in a clean format
                st.subheader(f"🇬🇧 English: {result_item['english_phrase']}")
                st.info(f"**Meaning:** {result_item['english_meaning']}")
                
                st.subheader(f"🇮🇳 Hindi (हिन्दी): {result_item['hindi_phrase']}")
                st.info(f"**Meaning (अर्थ):** {result_item['hindi_meaning']}")
                
                st.subheader(f"🇮🇳 Marathi (मराठी): {result_item['marathi_phrase']}")
                st.info(f"**Meaning (अर्थ):** {result_item['marathi_meaning']}")
                
            else:
                st.error("Sorry, I couldn't find a matching idiom in the database. Try rephrasing.")
        else:
            st.warning("Please enter an idiom to search.")

else:
    st.error("Application cannot start because the data file is missing.")