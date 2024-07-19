import os
import json
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import GraphQAChain
from vectorstore.search import SearchHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str
    use_kg: bool
    model_id: str

# Load the knowledge graph
graphml_path = "/mnt/data/GenerativeAIExamples/experimental/knowledge_graph_rag/backend/knowledge_graph.graphml"
if not os.path.exists(graphml_path):
    raise FileNotFoundError(f"Knowledge graph not found at {graphml_path}")

G = nx.read_graphml(graphml_path)
graph = NetworkxEntityGraph(G)

@app.get("/get-models/")
async def get_models():
    models = ChatNVIDIA.get_available_models()
    available_models = [model.id for model in models if model.model_type == "chat" and "instruct" in model.id]
    return {"models": available_models}

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    llm = ChatNVIDIA(model=request.model_id)
    graph_chain = GraphQAChain.from_llm(llm=llm, graph=graph, verbose=True)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    search_handler = SearchHandler("hybrid_demo3", use_bge_m3=True, use_reranker=True)

    user_input = request.user_input
    use_kg = request.use_kg
    response_data = {"user_input": user_input, "use_kg": use_kg}

    if use_kg:
        try:
            entity_string = llm.invoke("""Return a JSON with a single key 'entities' and list of entities within this user query. Each element in your list MUST BE part of the user's query. Do not provide any explanation. If the returned list is not parseable in Python, you will be heavily penalized. For example, input: 'What is the difference between Apple and Google?' output: ['Apple', 'Google']. Always follow this output format. Here's the user query: """ + user_input)
            entities = json.loads(entity_string.content)['entities']
            res = search_handler.search_and_rerank(user_input, k=5)
            context = "Here are the relevant passages from the knowledge base: \n\n" + "\n".join(item.text for item in res)
            all_triplets = []
            for entity in entities:
                all_triplets.extend(graph_chain.graph.get_entity_knowledge(entity, depth=2))
            context += "\n\nHere are the relationships from the knowledge graph: " + "\n".join(all_triplets)
            response_data["context"] = context
        except Exception as e:
            response_data["context"] = "No graph triples were available to extract from the knowledge graph. Always provide a disclaimer if you know the answer to the user's question, since it is not grounded in the knowledge you are provided from the graph."
    else:
        response_data["context"] = ""

    full_response = ""
    for response in chain.stream(f"Context: {response_data['context']}\n\nUser query: {user_input}" if use_kg else user_input):
        full_response += response
    response_data["assistant_response"] = full_response

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
