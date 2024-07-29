import os
import streamlit as st
import requests
import pandas as pd
import json

# Custom CSS to change heading font sizes
st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Evaluations")

st.subheader("Create synthetic Q&A pairs from large document chunks")

if 'documents' not in st.session_state:
    st.session_state['documents'] = None
if 'qa_pairs' not in st.session_state:
    st.session_state['qa_pairs'] = []
if 'evaluation_results' not in st.session_state:
    st.session_state['evaluation_results'] = []

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

    # Process Documents
    with st.container():
        st.markdown("### 1. Process Documents")
        if st.button("Process Documents"):
            res = has_pdf_files(directory)
            if not res:
                st.error("No PDF files found in directory! Only PDF files and text extraction are supported for now.")
                st.stop()
            progress_bar = st.progress(0)

            process_response = requests.post(
                "http://localhost:8000/evaluation/process-documents/",
                json={"directory": directory, "model_id": llm_selectbox}
            )
            if process_response.status_code == 200:
                st.session_state["documents"] = process_response.json().get("documents_processed")
                st.success(f"Finished splitting documents! Number of documents processed: {st.session_state['documents']}")
                progress_bar.progress(100)
            else:
                st.error("Error processing documents.")
                progress_bar.progress(0)

    # # Display processed document results
    # if st.session_state["documents"]:
    #     st.write(f"Processed documents: {st.session_state['documents']}")

    # Create Q&A pairs
    with st.container():
        st.markdown("### 2. Create synthetic Q&A pairs from large document chunks")
        if st.session_state["documents"] is not None:
            if st.button("Create Q&A pairs"):
                qa_placeholder = st.empty()
                json_list = []
                progress_bar = st.progress(0)
                try:
                    qa_response = requests.post(
                        "http://localhost:8000/evaluation/create-qa-pairs/",
                        json={"num_data": num_data, "model_id": llm_selectbox},
                        stream=True
                    )
                    if qa_response.status_code == 200:
                        total_lines = 0 
                        for line in qa_response.iter_lines():
                            if line:
                                try:
                                    pair = json.loads(line.decode('utf-8'))
                                    if 'question' in pair and 'answer' in pair:
                                        res = {
                                            'question': pair['question'],
                                            'answer': pair['answer']
                                        }
                                        st.write(res)
                                        json_list.append(res)
                                        total_lines += 1
                                        progress_bar.progress(min(total_lines / num_data, 1.0))  # Update progress
                                    else:
                                        st.error("Received data in an unexpected format.")
                                        st.write(pair)  # For debugging purposes
                                except json.JSONDecodeError:
                                    st.error("Error decoding JSON response.")
                        st.session_state['qa_pairs'] = json_list
                        st.success("Q&A pair generation completed.")
                        progress_bar.progress(100)
                    else:
                        st.error("Error creating Q&A pairs.")
                        progress_bar.progress(0)
                except requests.exceptions.ChunkedEncodingError as e:
                    st.error(f"Streaming error: {e}")
                    progress_bar.progress(0)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    progress_bar.progress(0)

    
    # Run Evaluation
    if os.path.exists("qa_data.csv"):
        with st.container():
            st.markdown("### 3. Load Q&A data and run evaluations of text vs graph vs text+graph RAG")
            if st.button("Run Evaluation"):
                df_csv = pd.read_csv("qa_data.csv")
                questions_list = df_csv["question"].tolist()
                answers_list = df_csv["answer"].tolist()
                eval_placeholder = st.empty()
                results = []
                progress_bar = st.progress(0)
                total_questions = len(questions_list)

                try:
                    eval_response = requests.post(
                        "http://localhost:8000/evaluation/run-evaluation/",
                        json={"questions_list": questions_list, "answers_list": answers_list, "model_id": llm_selectbox},
                        stream=True
                    )
                    if eval_response.status_code == 200:
                        for index, line in enumerate(eval_response.iter_lines()):
                            if line:
                                try:
                                    result = json.loads(line.decode('utf-8'))
                                    if 'question' in result and 'gt_answer' in result:
                                        results.append(result)
                                        st.session_state['evaluation_results'] = results
                                        # Update the displayed DataFrame
                                        results_df = pd.DataFrame(results)
                                        eval_placeholder.dataframe(results_df)
                                        progress_bar.progress(min((index + 1) / total_questions, 1.0))  # Update progress
                                    else:
                                        st.error("Received data in an unexpected format.")
                                        st.write(result)  # For debugging purposes
                                except json.JSONDecodeError:
                                    st.error("Error decoding JSON response.")
                        # Success message displayed after processing all lines
                        st.success("Combined results saved to 'combined_results.csv'")
                        progress_bar.progress(100)
                    else:
                        st.error("Error running evaluations.")
                except requests.exceptions.ChunkedEncodingError as e:
                    st.error(f"Streaming error: {e}")
                    progress_bar.progress(0)

                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    progress_bar.progress(0)


    # Run Scoring
    if os.path.exists("combined_results.csv"):
        with st.container():
            st.markdown("### 4. Run comparative evals for saved Q&A data")
            if st.button("Run Scoring"):
                combined_results = pd.read_csv("combined_results.csv").to_dict(orient="records")
                score_response = None
                score_placeholder = st.empty()
                results = []
                total_items = len(combined_results)
                progress_bar = st.progress(0)


                score_response = requests.post(
                    "http://localhost:8000/evaluation/run-scoring/",
                    json={"combined_results": combined_results}, 
                    stream=True
                )
                if score_response.status_code == 200:
                    for index,line in enumerate(score_response.iter_lines()):
                        if line:
                            try:
                                result = json.loads(line.decode('utf-8'))
                                if 'question' in result and 'gt_answer' in result:
                                    results.append(result)
                                    # Update the displayed DataFrame incrementally
                                    results_df = pd.DataFrame(results)
                                    score_placeholder.dataframe(results_df)
                                    progress_bar.progress(min((index + 1) / total_items, 1.0))  # Update progress
                                else:
                                    st.error("Received data in an unexpected format.")
                                    st.write(result)  # For debugging purposes
                            except json.JSONDecodeError:
                                st.error("Error decoding JSON response.")
                    # Success message displayed after processing all lines
                    st.success("Scoring completed and results saved to 'combined_results_with_scores.csv.")
                    # Save the final results to a CSV file
                    pd.DataFrame(results).to_csv('combined_results_with_scores.csv', index=False)
                    progress_bar.progress(100)
                else:
                    st.error("Error running scoring.")
                    progress_bar.progress(0)

if __name__ == "__main__":
    app()
