import requests


def rag_system(question, collection, document_type, n_chunks):
    results = collection.query(query_texts=[question],
    n_results=n_chunks,
    where={"doc_type": document_type} if document_type else None)
    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]
    context = ""
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        header = meta.get("header", "No header")
        context += f"{header}:\n{doc}\n\n"

    prompt = f"""
    You are a credit risk analyst who is designed to provide information about default risk for borrowers of a lending institution

   If the user asks about:
- Borrower-specific predictions (prompt involves specific information about a borrower) → Compare borrower to similar borrowers, summarize risk without exposing IDs.
- Model performance → report ONLY statistical metrics (e.g., KS, AUC, Brier Score) from the evaluation results provided to YOU, NOT individual borrowers. Use explanatory documents to understand what the statistical metrics mean and explain in simple terms.
- Data features → explain feature importance, Shapley values, or model coefficients.
- Reference explanatory documents to understand what each feature means and advise on how each feature influences risk and how risk can be reduced based on that info.

    Context:
    {context}

    Question:
    {question}

    Answer concisely and clearly. DO NOT EXPOSE SENSITIVE USER IDS
    
    """
    from openai import OpenAI
    import os
    import os
    import requests

    API_KEY = os.environ["MISTRAL_API_KEY"]
    URL = "https://api.mistral.ai/v1/chat/completions"

    def query_mistral(prompt, max_tokens=1000000, temperature=0.0):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Chat format: list of {role, content}
        payload = {
            "model": "mistral-small-latest",   # choose model available to you
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract the assistant's text
        return data["choices"][0]["message"]["content"]
    
    return query_mistral(prompt)