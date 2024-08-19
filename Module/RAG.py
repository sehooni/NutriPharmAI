import pandas as pd
import gradio as gr
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import BM25Retriever
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import openai as OpenAI
import os
import faiss
from langchain.llms import Predibase
import pandas as pd
from langchain.schema import Document
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_upstage import UpstageGroundednessCheck
from predibase import Predibase as pb
from kiwipiepy import Kiwi

def combine_text(row):
    # 각 컬럼의 내용을 결합하여 하나의 텍스트로 만듦
    return f"제품명: {row['이름']}\n영양성분: {row['성분']}\n기능: {row['기능']}\n주의사항: {row['주의사항']}"

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]



LANGCHAIN_TRACING_V2='true'
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="bisAI"

LANGCHAIN_API_KEY="..."
OPENAI_API_KEY = '...'
UPSTAGE_API_KEY ='...'
PREDIBASE_API_KEY = '...'

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PREDIBASE_API_KEY"] = PREDIBASE_API_KEY

kiwi = Kiwi()
upstage_ground_checker = UpstageGroundednessCheck()
OpenAIembedding = OpenAIEmbeddings()

csv_file_path = "./dataset/nutrient_data_final.csv"  # CSV 파일 경로
df = pd.read_csv(csv_file_path)

def combine_text(row):
    # 각 컬럼의 내용을 결합하여 하나의 텍스트로 만듦
    return f"제품명: {row['이름']}\n영양성분: {row['성분']}\n기능: {row['기능']}\n주의사항: {row['주의사항']}"

def mk_DB(df):
    documents = [Document(page_content=combine_text(row), metadata={"제품명": row['이름']}) for _, row in df.iterrows()]
    OpenAIembedding = OpenAIEmbeddings()

    embedding_model = OpenAIembedding
    if 'db' not in os.listdir() or ('index.faiss' not in os.listdir('./db') or 'index.pkl' not in os.listdir('./db')):
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local('./db')
    return documents

def LangGraph(pill, prompts):
    csv_file_path = "./dataset/nutrient_data_final.csv"  # CSV 파일 경로
    df = pd.read_csv(csv_file_path)
    documents = mk_DB(df=df)
    
    def make_retrieveQA_form(contexts,question):
        prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + contexts + "Question:" + question + "Helpful Answer: "
        return prompt

    def solar_generate_text(client, prompt, adapter_id, max_new_tokens):
        answer = client.generate(prompt, adapter_id=adapter_id, max_new_tokens=max_new_tokens).generated_text
        return answer

    # GraphState 상태를 저장하는 용도로 사용합니다.
    class GraphState(TypedDict):
        question: str  # 질문
        context: str  # 문서의 검색 결과
        answer: str  # 답변
        relevance: str  # 답변의 문서에 대한 관련성

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    def retrieve_document(state: GraphState) -> GraphState:
        # Question 에 대한 문서 검색을 retriever 로 수행합니다.
        retrieved_docs = retriever.invoke(state["question"]) ###
        # 검색된 문서를 context 키에 저장합니다.
        return GraphState(context= retrieved_docs) ###

    #Solar Chatbot model을 사용하여 답변을 생성합니다.
    def llm_answer(state: GraphState) -> GraphState:
        if len(query.split(',')) > 0:
            context_n = len(query.split(','))+1
            state['context'] = state['context'][:context_n]
        else:
            state['context'] = state['context']
        contexts = ", ".join([state['context'][i].page_content for i in range(len(state['context']))])
        prompt = make_retrieveQA_form(contexts,state['question'])
        return GraphState(
            answer=solar_generate_text(client, prompt, adapter_id, max_new_tokens),
            context=state["context"],
            question=state["question"],
        )
        
    # Upstage Ground Checker로 관련성 체크를 실행합니다.
    def relevance_check(state: GraphState) -> GraphState:
        # 관련성 체크를 실행합니다. 결과: grounded, notGrounded, notSure
        response = upstage_ground_checker.run(
            {"context": state["context"], "answer": state["answer"]}
        )
        return GraphState(
            relevance=response,
            context=state["context"],
            answer=state["answer"],
            question=state["question"],
        )

    # 관련성 체크 결과를 반환합니다.
    def is_relevant(state: GraphState) -> GraphState:
        if state["relevance"] == "grounded":
            return "관련성 O"
        elif state["relevance"] == "notGrounded":
            return "관련성 X"
        elif state["relevance"] == "notSure":
            return "확인불가"

    solar_llm = Predibase(model="solar-1-mini-chat-240612", 
                    predibase_api_key=os.environ["PREDIBASE_API_KEY"], 
                    adapter_id="AIMedicine", 
                    adapter_version=1,
                    max_new_tokens = 4096)

    gpt4o_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

    embedding_model = OpenAIembedding
    vector_db = FAISS.load_local("./db", embedding_model, allow_dangerous_deserialization=True)
    index = faiss.read_index("./db/index.faiss")

    MultiQuery_retriever = MultiQueryRetriever.from_llm(llm=gpt4o_llm,retriever=vector_db.as_retriever())
    bm25_retriever = BM25Retriever.from_documents(documents)

    kiwi_bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=kiwi_tokenize)
    faiss_retriever = FAISS.from_documents(documents, embedding_model).as_retriever()

    MultiQuery_bm25_73 = EnsembleRetriever(
        retrievers=[MultiQuery_retriever, bm25_retriever],  # 리트리버
        weights=[0.7, 0.3],
        search_type="mmr"
    )

    kiwibm25_faiss_73 = EnsembleRetriever(
        retrievers=[kiwi_bm25_retriever, faiss_retriever],  # 사용할 검색 모델의 리스트
        weights=[0.7, 0.3],  # 각 검색 모델의 결과에 적용할 가중치
        search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
    )

    retriever = kiwibm25_faiss_73
    # RetrievalQA Chain 설정
    gpt4o_qa_chain = RetrievalQA.from_chain_type(
        llm=gpt4o_llm,
        chain_type="stuff",
        retriever=retriever,
    )

    solar_qa_chain = RetrievalQA.from_chain_type(
        llm=solar_llm,
        chain_type="stuff",
        retriever=retriever,
    )

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_document)  # 에이전트 노드를 추가합니다.
    workflow.add_node("llm_answer", llm_answer)  # 정보 검색 노드를 추가합니다.
    workflow.add_node("relevance_check", relevance_check)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.

    workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
    workflow.add_edge("llm_answer", "relevance_check")  # 답변 -> 관련성 체크
    workflow.add_conditional_edges(
        "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        is_relevant,
        {
            "관련성 O": END,  # 관련성이 있으면 종료합니다.
            "관련성 X": "retrieve",  # 관련성이 없으면 다시 답변을 생성합니다.
            "확인불가": "retrieve",  # 관련성 체크 결과가 모호하다면 다시 답변을 생성합니다.
        },
    )

    workflow.set_entry_point("retrieve")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # recursion_limit: 최대 반복 횟수, thread_id: 실행 ID (구분용)
    config = RunnableConfig(recursion_limit=13, configurable={"thread_id": "SELF-RAG"})
    
    query = pill
    
    pb_token = pb(api_token=PREDIBASE_API_KEY)
    client = pb_token.deployments.client("solar-1-mini-chat-240612")
    adapter_id, max_new_tokens ="AIMedicine/1", 5000

    inputs = GraphState(question=prompts)
    qa_chain = solar_qa_chain  # solar_qa_chain or gpt4o_qa_chain
    output = app.invoke(inputs, config=config)
    
    return output['answer']

