# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import ast
import networkx as nx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import GraphQAChain
from vectorstore.search import SearchHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from utils.langsmith_setup import get_langsmith_callbacks, get_langsmith_run_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define a Pydantic model for the chat request body
class ChatRequest(BaseModel):
    user_input: str
    use_kg: bool
    model_id: str


def _extract_entities_from_llm_output(raw_output: str) -> list[str]:
    if not raw_output:
        return []

    parsed = None
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_output)
        except (ValueError, SyntaxError):
            return []

    if isinstance(parsed, dict):
        entities = parsed.get("entities", [])
    elif isinstance(parsed, list):
        entities = parsed
    else:
        return []

    if not isinstance(entities, list):
        return []

    cleaned_entities = []
    for entity in entities:
        if isinstance(entity, str):
            stripped = entity.strip()
            if stripped:
                cleaned_entities.append(stripped)
    return cleaned_entities


def _find_entities_in_query(query: str, graph_nodes, limit: int = 8) -> list[str]:
    query_lower = query.lower()
    matched_entities = []
    for candidate in graph_nodes:
        if isinstance(candidate, str) and candidate.lower() in query_lower:
            matched_entities.append(candidate)
            if len(matched_entities) >= limit:
                return list(dict.fromkeys(matched_entities))
    return list(dict.fromkeys(matched_entities))


def _get_data_dir() -> str:
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return data_dir
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    logger.warning("DATA_DIR not set; defaulting to %s", default_dir)
    return default_dir


def _get_collection_name() -> str:
    return os.getenv("MILVUS_COLLECTION_NAME", "hybrid_demo3")

# Define an endpoint to get available models
@router.get("/get-models/")
async def get_models():
    models = ChatNVIDIA.get_available_models()
    available_models = [model.id for model in models if model.model_type == "chat" and "instruct" in model.id]
    return {"models": available_models}

# Define an endpoint for the chat interface
@router.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    response_data = {"user_input": request.user_input, "use_kg": request.use_kg}
    run_config = get_langsmith_run_config(tags=["chat"], metadata={"use_kg": request.use_kg})
    callbacks = get_langsmith_callbacks()
    llm = ChatNVIDIA(model=request.model_id, callbacks=callbacks)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
    )
    chain = prompt_template | llm | StrOutputParser()


    user_input = request.user_input
    search_handler = SearchHandler(_get_collection_name(), use_bge_m3=True, use_reranker=True)
    res = search_handler.search_and_rerank(user_input, k=5)
    context = "Here are the relevant passages from the knowledge base: \n\n" + "\n".join(item.text for item in res)

    if request.use_kg:
        data_dir = _get_data_dir()
        KG_GRAPHML_PATH = os.path.join(data_dir, "knowledge_graph.graphml")

        logger.info(f"Entering {KG_GRAPHML_PATH}")
        if os.path.exists(KG_GRAPHML_PATH):
            G = nx.read_graphml(KG_GRAPHML_PATH)
            graph = NetworkxEntityGraph(G)
            graph_available = True
        else:
            logger.error(f"Knowledge graph not found at {KG_GRAPHML_PATH}")
            graph_available = False

        if not graph_available:
            return {"assistant_response": "The knowledge graph is currently unavailable. Please try again later."}

        llm = ChatNVIDIA(model=request.model_id, callbacks=callbacks)
        graph_chain = GraphQAChain.from_llm(llm=llm, graph=graph, verbose=True, callbacks=callbacks)

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
        )
        chain = prompt_template | llm | StrOutputParser()

        try:
            entity_string = llm.invoke(
                """Return only entities from the user query in strict JSON format.
JSON schema: {"entities": ["entity1", "entity2"]}
Rules:
1) Every entity must be an exact span from the user query.
2) Return only the JSON object, no extra text.
3) If there are no entities, return {"entities": []}.
User query: """
                + user_input,
                config=run_config,
            )
            entities_from_llm = _extract_entities_from_llm_output(entity_string.content)
            entities_from_query = _find_entities_in_query(user_input, G.nodes())
            entities = list(dict.fromkeys(entities_from_llm + entities_from_query))
            all_triplets = []
            for entity in entities:
                all_triplets.extend(graph_chain.graph.get_entity_knowledge(entity, depth=2))
            all_triplets = list(dict.fromkeys(all_triplets))
            logger.info("KG query enabled. entities=%s, triplets_found=%s", entities, len(all_triplets))
            context += "\n\nHere are the relationships from the knowledge graph: " + "\n".join(all_triplets)
        except Exception as e:
            logger.exception("Knowledge graph retrieval failed: %s", str(e))
            context += "\n\nNo graph triples were available to extract from the knowledge graph. Always provide a disclaimer if you know the answer to the user's question, since it is not grounded in the knowledge you are provided from the graph."

    response_data["context"] = context

    full_response = llm.invoke(
        f"Context: {response_data['context']}\n\nUser query: {request.user_input}",
        config=run_config,
    )
    response_data["assistant_response"] = full_response if isinstance(full_response, str) else full_response.content

    return response_data
