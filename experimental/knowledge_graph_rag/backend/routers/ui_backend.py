import os
import pandas as pd
import networkx as nx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from utils.lc_graph import process_documents, save_triples_to_csvs
from vectorstore.search import SearchHandler
from langchain_nvidia_ai_endpoints import ChatNVIDIA

router = APIRouter()

class DirectoryRequest(BaseModel):
    directory: str
    model_id: str

@router.get("/get-models/")
async def get_models():
    models = ChatNVIDIA.get_available_models()
    available_models = [model.id for model in models if model.model_type == "chat" and "instruct" in model.id]
    return {"models": available_models}

@router.post("/process-documents/")
async def process_documents_endpoint(request: DirectoryRequest, background_tasks: BackgroundTasks):
    directory = request.directory
    model_id = request.model_id
    llm = ChatNVIDIA(model=model_id)

    # Save progress updates in a temporary file
    progress_file = "progress.txt"
    with open(progress_file, "w") as f:
        f.write("0")

    def update_progress(completed_futures, total_futures):
        progress = completed_futures / total_futures
        with open(progress_file, "w") as f:
            f.write(str(progress))

    def background_task():
        documents, results = process_documents(directory, llm, update_progress=update_progress)
        search_handler = SearchHandler("hybrid_demo3", use_bge_m3=True, use_reranker=True)
        search_handler.insert_data(documents)
        save_triples_to_csvs(results)
        
        triples_df = pd.read_csv("triples.csv")
        entities_df = pd.read_csv("entities.csv")
        relations_df = pd.read_csv("relations.csv")

        entity_name_map = entities_df.set_index("entity_id")["entity_name"].to_dict()
        relation_name_map = relations_df.set_index("relation_id")["relation_name"].to_dict()

        G = nx.from_pandas_edgelist(
            triples_df,
            source="entity_id_1",
            target="entity_id_2",
            edge_attr="relation_id",
            create_using=nx.DiGraph,
        )

        G = nx.relabel_nodes(G, entity_name_map)
        edge_attributes = nx.get_edge_attributes(G, "relation_id")

        new_edge_attributes = {
            (u, v): relation_name_map[edge_attributes[(u, v)]]
            for u, v in G.edges()
            if edge_attributes[(u, v)] in relation_name_map
        }
        nx.set_edge_attributes(G, new_edge_attributes, "relation")

        nx.write_graphml(G, "knowledge_graph.graphml")

    background_tasks.add_task(background_task)
    return {"message": "Processing started"}

@router.get("/progress/")
async def get_progress():
    try:
        with open("progress.txt", "r") as f:
            progress = f.read()
        return {"progress": progress}
    except FileNotFoundError:
        return {"progress": "0"}
