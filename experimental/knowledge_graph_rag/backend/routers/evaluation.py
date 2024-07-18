import os
import random
import pandas as pd
import networkx as nx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import GraphQAChain
from vectorstore.search import SearchHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from utils.preprocessor import generate_qa_pair
from utils.lc_graph import process_documents, save_triples_to_csvs
from llama_index.core import SimpleDirectoryReader
from openai import OpenAI

router = APIRouter()

class ProcessRequest(BaseModel):
    directory: str
    model_id: str

class QAPairsRequest(BaseModel):
    num_data: int
    model_id: str

class QARequest(BaseModel):
    questions_list: list
    answers_list: list
ƒ
class ScoreRequest(BaseModel):
    combined_results: list

reward_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)
@router.get("/get-models/")
async def get_models():
    models = ChatNVIDIA.get_available_models()
    available_models = [model.id for model in models if model.model_type == "chat" and "instruct" in model.id]
    return {"models": available_models}

def load_data(input_dir, num_workers):
    reader = SimpleDirectoryReader(input_dir=input_dir)
    documents = reader.load_data(num_workers=num_workers)
    return documents

def get_reward_scores(question, answer):
    completion = reward_client.chat.completions.create(
        model="nvidia/nemotron-4-340b-reward",
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    )
    try:
        content = completion.choices[0].message[0].content
        res = content.split(",")
        content_dict = {}
        for item in res:
            name, val = item.split(":")
            content_dict[name] = float(val)
        return content_dict
    except:
        return None

def process_question(question, answer, llm):
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_text = executor.submit(get_text_RAG_response, question, llm)
        future_graph = executor.submit(get_graph_RAG_response, question, llm)
        future_combined = executor.submit(get_combined_RAG_response, question, llm)

        text_RAG_response = future_text.result()
        graph_RAG_response = future_graph.result()
        combined_RAG_response = future_combined.result()

    return {
        "question": question,
        "gt_answer": answer,
        "textRAG_answer": text_RAG_response,
        "graphRAG_answer": graph_RAG_response,
        "combined_answer": combined_RAG_response
    }

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
)

def get_text_RAG_response(question, llm):
    chain = prompt_template | llm | StrOutputParser()
    search_handler = SearchHandler("hybrid_demo3", use_bge_m3=True, use_reranker=True)
    res = search_handler.search_and_rerank(question, k=5)
    context = "Here are the relevant passages from the knowledge base: \n\n" + "\n".join(item.text for item in res)
    answer = chain.invoke("Context: " + context + "\n\nUser query: " + question)
    return answer

def get_graph_RAG_response(question, llm):
    chain = prompt_template | llm | StrOutputParser()
    entity_string = llm.invoke("""Return a JSON with a single key 'entities' and list of entities within this user query. Each element in your list MUST BE part of the user's query. Do not provide any explanation. If the returned list is not parseable in Python, you will be heavily penalized. For example, input: 'What is the difference between Apple and Google?' output: ['Apple', 'Google']. Always follow this output format. Here's the user query: """ + question)
    G = nx.read_graphml(os.path.join("data", "knowledge_graph.graphml"))
    graph = NetworkxEntityGraph(G)

    try:
        entities = json.loads(entity_string.content)['entities']
        context = ""
        all_triplets = []
        for entity in entities:
            all_triplets.extend(graph.get_entity_knowledge(entity, depth=2))
        context = "Here are the relationships from the knowledge graph: " + "\n".join(all_triplets)
    except:
        context = "No graph triples were available to extract from the knowledge graph. Always provide a disclaimer if you know the answer to the user's question, since it is not grounded in the knowledge you are provided from the graph."
    answer = chain.invoke("Context: " + context + "\n\nUser query: " + question)
    return answer

def get_combined_RAG_response(question, llm):
    chain = prompt_template | llm | StrOutputParser()
    entity_string = llm.invoke("""Return a JSON with a single key 'entities' and list of entities within this user query. Each element in your list MUST BE part of the user's query. Do not provide any explanation. If the returned list is not parseable in Python, you will be heavily penalized. For example, input: 'What is the difference between Apple and Google?' output: ['Apple', 'Google']. Always follow this output format. Here's the user query: """ + question)
    G = nx.read_graphml(os.path.join("data", "knowledge_graph.graphml"))
    graph = NetworkxEntityGraph(G)

    try:
        entities = json.loads(entity_string.content)['entities']
        search_handler = SearchHandler("hybrid_demo3", use_bge_m3=True, use_reranker=True)
        res = search_handler.search_and_rerank(question, k=5)
        context = "Here are the relevant passages from the knowledge base: \n\n" + "\n".join(item.text for item in res)
        all_triplets = []
        for entity in entities:
            all_triplets.extend(graph.get_entity_knowledge(entity, depth=2))
        context += "\n\nHere are the relationships from the knowledge graph: " + "\n".join(all_triplets)
    except Exception as e:
        context = "No graph triples were available to extract from the knowledge graph. Always provide a disclaimer if you know the answer to the user's question, since it is not grounded in the knowledge you are provided from the graph."
    answer = chain.invoke("Context: " + context + "\n\nUser query: " + question)
    return answer

@router.post("/process-documents/")
async def process_documents_endpoint(request: ProcessRequest, background_tasks: BackgroundTasks):
    directory = request.directory
    model_id = request.model_id
    llm = ChatNVIDIA(model=model_id)

    documents, results = process_documents(directory, llm, triplets=False, chunk_size=2000, chunk_overlap=200)
    return {"message": "Document processing started", "documents_processed": len(documents)}

@router.post("/create-qa-pairs/")
async def create_qa_pairs(request: QAPairsRequest):
    print("entered")
    num_data = request.num_data
    model_id = request.model_id
    llm = ChatNVIDIA(model=model_id)

    if not os.path.exists('documents.csv'):
        raise HTTPException(status_code=404, detail="Documents not found. Please process documents first.")
  
    df = pd.read_csv('documents.csv')
    documents = [SimpleDirectoryReader.from_dict(row) for index, row in df.iterrows()]
    json_list = []
    
    qa_docs = random.sample(documents, num_data)
    for doc in qa_docs:
        res = generate_qa_pair(doc, llm)
        if res:
            json_list.append(res)
    
    if len(json_list) > 0:
        qa_df = pd.DataFrame(json_list)
        qa_df.to_csv('qa_data.csv', index=False)
    else:
        raise HTTPException(status_code=500, detail="No Q&A pairs generated")

    return {"message": "Q&A pairs created"}

@router.post("/run-evaluation/")
async def run_evaluation(request: QARequest):
    questions_list = request.questions_list
    answers_list = request.answers_list
    llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")  # or any other default model

    results = []
    for question, answer in zip(questions_list, answers_list):
        result = process_question(question, answer, llm)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("combined_results.csv", index=False)
    return {"message": "Evaluation completed and results saved"}

@router.post("/run-scoring/")
async def run_scoring(request: ScoreRequest):
    combined_results = request.combined_results

    score_columns = ['gt', 'textRAG', 'graphRAG', 'combinedRAG']
    metrics = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']

    for row in combined_results:
        res_gt = get_reward_scores(row["question"], row["gt_answer"])
        res_textRAG = get_reward_scores(row["question"], row["textRAG_answer"])
        res_graphRAG = get_reward_scores(row["question"], row["graphRAG_answer"])
        res_combinedRAG = get_reward_scores(row["question"], row["combined_answer"])

        for score_type, res in zip(score_columns, [res_gt, res_textRAG, res_graphRAG, res_combinedRAG]):
            for metric in metrics:
                row[f'{score_type}_{metric}'] = res[metric]

    df = pd.DataFrame(combined_results)
    df.to_csv("combined_results_with_scores.csv", index=False)
    return {"message": "Scoring completed and results saved"}
