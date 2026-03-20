### Credit Risk Chatbot

## Overview

This repository contains source code for a credit risk chatbot. The chatbot is able to summarize risk for new borrowers by aggregating risk data from historical borrowers, the chatbot is also able to interpret ML models used for default predictions, and is able to provide suggestions to reduce credit risk.

## Credit Risk Dataset

The credit risk dataset used for this project is simulated based on known macroeconomic relationships. For example, income rises with age but drops after reaching a peak age. FICO score also depends heavily on credit utilization rate which mimics real world scoring. The file simulate.py simulates a dataset with a customizable number of borrowers, and saves this data to a table within in SQL database.

## Modelling Probability of Default (Logistic)

The first ML model used for default forecasting is a logistic regression. This model is trained on raw data from the SQL database. Predicted default probabilities are added to the table containing raw credit risk data in the database. A new table with the logistic regression model summary (coefficients) is also created. A model metrics table is also created to track KS, AUC, and Brier Scores for models. I also bin the data and use weight of evidence scoring to build a new logistic model. The database tables are updated with new predictions, model summaries, and performance metrics.

## Modelling Probability of Default (XGBoost)

The next ML model used for default forecasting is a XGBoost model. This model is trained on raw feature data as WOE binning is not required for tree-based models. The performance metrics are tracked in the SQL database. A new table is also created to provide mean aggregated shapley values for each raw feature.

## Vector Database

The SQL database is converted to a vector database. For each table, a markdown document is generated that explains all the information captured in the table. Raw data from SQL tables are then loaded into a ChromaDB vector database.

## Query Routing

There are four types of queries a user can provide to the chatbot: Queries concerning new borrowers, Queries concerning model performance, Queries concerning factors that drive delinquency for Logistic/Tree-based models. A Naive bayes classifier is used to classify which type of query is entered into the chatbot so the documents searched during RAG can be appropriately filtered.

## RAG/LLM

After the documents are filtered, RAG finds the eight nearest neighbors to the provided chunk and then that information along with the chat history is provided as context to the MistralAI large language model. This model can then answer user questions.

## User Interface

The user interface is hosted on streamlit where users can chat with the large language model and receive personalized credit risk insights.

## Deployment

The app is hosted on an AWS EC2 instance, the query classification model and vector database are stored within an AWS S3 bucket that can be accessed from the EC2 instance. The link to the application is here: http://18.116.231.26:8501/
