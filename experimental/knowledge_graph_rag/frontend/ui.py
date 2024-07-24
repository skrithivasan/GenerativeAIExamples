import os
import streamlit as st
import requests
import time

st.title("Knowledge Graph RAG")
st.subheader("Load Data from Files")

if 'documents' not in st.session_state:
    st.session_state['documents'] = None

# Fetch available models from the backend
response = requests.get("http://localhost:8000/ui/get-models/")
if response.status_code == 200:
    available_models = response.json()["models"]
else:
    st.error("Error fetching models.")
    available_models = []

with st.sidebar:
    llm = st.selectbox("Choose an LLM", available_models, index=available_models.index("mistralai/mixtral-8x7b-instruct-v0.1") if "mistralai/mixtral-8x7b-instruct-v0.1" in available_models else 0)
    st.write("You selected: ", llm)

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
        
        response = requests.post("http://localhost:8000/ui/process-documents/", json={"directory": directory, "model_id": llm})
        if response.status_code == 200:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            while True:
                progress_response = requests.get("http://localhost:8000/ui/progress/")
                if progress_response.status_code == 200:
                    try:
                        progress_data = progress_response.json()
                        progress = float(progress_data.get("progress", 0))  # Default to 0 if not found
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing: {int(progress * 100)}%")
                        if progress >= 1.0:
                            break
                    except (ValueError, KeyError):
                        st.error("Received invalid progress data.")
                        break
                else:
                    st.error("Error fetching progress data.")
                    break
                time.sleep(1)


            st.success("Processing complete and graph created. Saved to knowledge_graph.graphml")
        else:
            st.error("Error processing documents.")

if __name__ == "__main__":
    app()
