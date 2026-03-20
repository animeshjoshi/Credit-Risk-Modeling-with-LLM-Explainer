# app.py
import streamlit as st
from RAG import rag_system
import chromadb

# -----------------------------
# Load Chroma collection
# -----------------------------
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="credit_risk_db")
    collection = client.get_or_create_collection(name="credit_risk")
    return collection

collection = load_collection()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Credit Risk Chatbot")
st.title("Welcome to the Credit Risk Chatbot")
st.write("Ask questions about credit risk, features, or borrowers.")

# -----------------------------
# Initialize chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Clear chat button
# -----------------------------
if st.button("Clear Chat"):
    st.session_state.chat_history = []

# -----------------------------
# Display previous chat messages
# -----------------------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Type your question here")

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):

        # Build conversation context for memory
        conversation = ""
        for chat in st.session_state.chat_history:
            conversation += f"User: {chat['question']}\nAssistant: {chat['answer']}\n"

        conversation += f"User: {user_input}\nAssistant:"

        # Call RAG system
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib
        clf = joblib.load("models/query_classifier.pkl")
        vectorizer = joblib.load("models/query_vectorizer.pkl")
        X_new = vectorizer.transform([user_input])
        predicted_label = clf.predict(X_new)[0]
        answer = rag_system(conversation, collection, document_type=predicted_label, n_chunks=8)

        # Store Q&A in chat history
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer
        })

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)