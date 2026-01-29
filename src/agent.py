import os
from typing import List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langgraph.graph import END, StateGraph

# --- CONFIGURATION ---
# Fix paths relative to where this script is imported (usually root)
DB_DIR = os.path.join(os.getcwd(), "data", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3"

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    reformulated_count: int

def get_graph():
    """
    Initializes and returns the compiled LangGraph workflow.
    """
    
    # 1. Setup Components
    print("Loading Embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    print("Loading Vector Store...")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    retriever = vector_store.as_retriever()
    
    print("Loading LLM...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json") # For grading
    llm_gen = ChatOllama(model=LLM_MODEL, temperature=0) # For generation

    # 2. Define Nodes
    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        
        prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning useful to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination."""
        )
        chain = prompt | llm | JsonOutputParser()
        
        filtered_docs = []
        for d in documents:
            try:
                score = chain.invoke({"question": question, "context": d.page_content})
                grade = score["score"]
                if grade == "yes":
                    filtered_docs.append(d)
            except Exception:
                continue
                
        return {"documents": filtered_docs, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert research assistant. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer technical, professional, and concise.
            
            Question: {question} 
            Context: {context} 
            Answer:"""
        )
        chain = prompt | llm_gen | StrOutputParser()
        generation = chain.invoke({"context": documents, "question": question})
        return {"generation": generation}

    def transform_query(state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        count = state.get("reformulated_count", 0) + 1
        
        if count > 3:
            return {"question": question, "reformulated_count": count}

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helper that re-writes questions to improve retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question. Output only the improved question string.")
        ])
        chain = prompt | llm_gen | StrOutputParser()
        better_question = chain.invoke({"question": question})
        return {"question": better_question, "reformulated_count": count}

    # 3. Define Conditional Logic
    def decide_to_generate(state):
        filtered_documents = state["documents"]
        count = state.get("reformulated_count", 0)
        
        if not filtered_documents and count <= 3:
            return "transform_query"
        else:
            return "generate"

    # 4. Build Graph
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
