doc_types = {
    "Model Explanations": "These are questions about the methodology used for modeling the probability of default. Questions could involve the advantages/disadvantages of Weight of Evidence transformation, methodological differences between regressions and tree-based methods, tradeoffs between performance and interpretability.",
    "Logistic Regression Interpretation": "Questions about how the logistic regression credit risk models are interpreted (i.e what does the coefficient represent, what does p-value represent)",
    "Logistic Model Performance Metrics": "Questions about how accuracy and performance of models is measured (i.e How can we tell that logistic regression performs well)",
    "Model Performance Metrics": "Questions about model performance like AUC, KS, Brier Score (i.e Does logistic regression outperform tree-based methods, are models accurate in forecasting default risk, can a bank trust these methods)",
    "Shapley Value Interpretation": "Questions about interpreting shapley values (what are shapley values and how do they help us understand default predictions)",
    "Data Interpretation": "Questions about the data used to model credit risk. Example questions would be what features are used, what is LTV, what is a fico score, what is a risk rating",
    "Loan Level Data": "Questions about specific borrowers and their credit risk. Must provide specific information about borrowers such as FICO, LTV, Income, Employment Status, Debt, etc.",
    "Logistic Model Configs": "Questions about relationships between features and credit risk uncovered by logistic regression. Questions include: How is FICO related to credit risk, How do FICO and Risk Rating interact and contribute to default probability?, Role of LTV in risk?",
    "Shapley Values Computed": "Questions about relationships between features and credit risk uncovered by XGBoost. Questions include: How is FICO related to credit risk, How do FICO and Risk Rating interact and contribute to default probability?, Role of LTV in risk?",

}

import json
import os 
import json
import os
import requests
import json

API_KEY = os.environ["MISTRAL_API_KEY"]
URL = "https://api.mistral.ai/v1/chat/completions"


def query_mistral(prompt, max_tokens=25600, temperature=0.3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


training_data = []

for label, description in doc_types.items():

    prompt = f"""
Generate 50 realistic user questions about:

{description}

Rules:
- Mix short and long queries
- Include both casual and technical language
- Return ONLY a JSON list of questions
"""

    #response_text = query_mistral(prompt)
    response_text = query_mistral(prompt, max_tokens=2500)

    # remove markdown wrappers
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        queries = json.loads(response_text)
    except Exception as e:
        print("JSON parsing failed, raw output:")
        print(response_text)
        continue

    for q in queries:
        training_data.append({
            "query": q,
            "label": label
        })

print("Total queries generated:", len(training_data))

with open("query_training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("Saved training data to query_training_data.json")