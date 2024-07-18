import os
import streamlit as st
import requests
import random
import pandas as pd
import time

st.title("Evaluations")

st.subheader("Create synthetic Q&A pairs from large document chunks")

if 'documents' not in st.session_state:
    st.session_state['documents'] = None

response = requests.get("http://localhost:8000/evaluation/get-models/")
if response.status_code == 200:
    available_models = response.json()["models"]
else:
    st.error("Error fetching models.")
    available_models = []

with st.sidebar:
    llm_selectbox = st.selectbox("Choose an LLM", available_models, index=available_models.index("mistralai/mixtral-8x7b-instruct-v0.1") if "mistralai/mixtral-8x7b-instruct-v0.1" in available_models else 0)
    st.write("You selected: ", llm_selectbox)

    num_data = st.slider("How many Q&A pairs to generate?", 10, 100, 50, step=10)

def has_pdf_files(directory):
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            return True
    return False

def app():
    cwd = os.getcwd()
    directories = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d)) and not d.startswith('.') and '__' not in d]
    selected_dir = st.selectbox("Select a directory:", directories, index=0)
    directory = os.path.join(cwd, selected_dir)

    if st.button("Process Documents"):
        res = has_pdf_files(directory)
        if not res:
            st.error("No PDF files found in directory! Only PDF files and text extraction are supported for now.")
            st.stop()

        process_response = requests.post(
            "http://localhost:8000/evaluation/process-documents/",
            json={"directory": directory, "model_id": llm_selectbox}
        )
        if process_response.status_code == 200:
            st.session_state["documents"] = process_response.json().get("documents_processed")
            st.success("Finished splitting documents!")
        else:
            st.error("Error processing documents.")

    if st.button("Create Q&A pairs"):
        qa_response = requests.post(
            "http://localhost:8000/evaluation/create-qa-pairs/",
            json={"num_data": num_data, "model_id": llm_selectbox}
        )
        if qa_response.status_code == 200:
            st.success("Q&A pairs created.")
        else:
            st.error("Error creating Q&A pairs.")

    if os.path.exists("qa_data.csv"):
        with st.expander("Load Q&A data and run evaluations of text vs graph vs text+graph RAG"):
            if st.button("Run"):
                df_csv = pd.read_csv("qa_data.csv")
                questions_list = df_csv["question"].tolist()
                answers_list = df_csv["answer"].tolist()

                eval_response = requests.post(
                    "http://localhost:8000/evaluation/run-evaluation/",
                    json={"questions_list": questions_list, "answers_list": answers_list}
                )
                if eval_response.status_code == 200:
                    st.success("Evaluation completed and results saved.")
                else:
                    st.error("Error running evaluations.")

    if os.path.exists("combined_results.csv"):
        with st.expander("Run comparative evals for saved Q&A data"):
            if st.button("Run scoring"):
                combined_results = pd.read_csv("combined_results.csv").to_dict(orient="records")

                score_response = requests.post(
                    "http://localhost:8000/evaluation/run-scoring/",
                    json={"combined_results": combined_results}
                )
                if score_response.status_code == 200:
                    st.success("Scoring completed and results saved.")
                else:
                    st.error("Error running scoring.")

                combined_results_df = pd.read_csv("combined_results_with_scores.csv")
                st.write("First few rows of the updated data:")
                st.dataframe(combined_results_df.head())

if __name__ == "__main__":
    app()
