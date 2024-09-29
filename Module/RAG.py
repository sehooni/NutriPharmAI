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
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_upstage import UpstageGroundednessCheck
from predibase import Predibase as pb
from kiwipiepy import Kiwi
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage


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

# CSV 파일 경로
sup_product_df = pd.read_csv("./dataset/nutrient_data_final.csv")
med_product_df = pd.read_csv("./dataset/new_medi.csv")
med_info_df = pd.read_csv("./dataset/pdf_contents.csv")
# csv_file_path = "./dataset/nutrient_data_final.csv"  # CSV 파일 경로
# df = pd.read_csv(csv_file_path)

def combine_sup_product(row):
    # 각 컬럼의 내용을 결합하여 하나의 텍스트로 만듦
    return f"제품명: {row['이름']}\n영양성분: {row['성분']}\n기능: {row['기능']}\n주의사항: {row['주의사항']}"
def combine_pdf(row):
    return f"제목: {row['제목']}\n요약: {row['요약']}\n내용: {row['내용']}"
def combine_med_product(row):
    return f"제품명: {row['제품명']}\n주성분: {row['주성분']}\n기능: {row['이 약의 효능은 무엇입니까?']}\n주의사항: {row['이 약의 사용상 주의사항은 무엇입니까?']}"

# def mk_DB(sup_product_df, med_product_df, med_info_df):
    # documents = [Document(page_content=combine_text(row), metadata={"제품명": row['이름']}) for _, row in df.iterrows()]
    # documents_sup_product = [Document(page_content=combine_sup_product(row), metadata={"제품명": row['이름']}) for _, row in sup_product_df.iterrows()]
    # documents_medi_product = [Document(page_content=combine_med_product(row), metadata={"제품명": row['제품명']}) for _, row in med_product_df.iterrows()]
    # documents_medi_info = [Document(page_content=combine_pdf(row), metadata={"제목": row['제목']}) for _, row in med_info_df.iterrows()]

    # OpenAIembedding = OpenAIEmbeddings()
    # SolarEmbedding = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

    # embedding_model = OpenAIembedding
    #embedding_model = SolarEmbedding
    # if 'db' not in os.listdir() or ('index.faiss' not in os.listdir('./db') or 'index.pkl' not in os.listdir('./db')):
    #     vector_store = FAISS.from_documents(documents, embedding_model)
    #     vector_store.save_local('./db')
    # return documents

def retriever_medi(documents, embedding_model):
    vector_store = FAISS.from_documents(documents, embedding_model)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    kiwi_bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=kiwi_tokenize)
    
    kiwibm25_faiss_73 = EnsembleRetriever(
    retrievers=[kiwi_bm25_retriever, faiss_retriever],  # 사용할 검색 모델의 리스트
    weights=[0.7, 0.3],  # 각 검색 모델의 결과에 적용할 가중치
    search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
    )
    return faiss_retriever

def retriever_sup(documents, embedding_model):
    vector_store = FAISS.from_documents(documents, embedding_model)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    kiwi_bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=kiwi_tokenize)
    
    kiwibm25_faiss_73 = EnsembleRetriever(
    retrievers=[kiwi_bm25_retriever, faiss_retriever],  # 사용할 검색 모델의 리스트
    weights=[0.7, 0.3],  # 각 검색 모델의 결과에 적용할 가중치
    search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
    )
    return kiwibm25_faiss_73



def LangGraph(pill, prompts, chat_type='medi'):
    # csv_file_path = "./dataset/nutrient_data_final.csv"  # CSV 파일 경로
    # df = pd.read_csv(csv_file_path)
    # documents = mk_DB(df=df)
    documents_sup_product = [Document(page_content=combine_sup_product(row), metadata={"제품명": row['이름']}) for _, row in sup_product_df.iterrows()]
    documents_medi_product = [Document(page_content=combine_med_product(row), metadata={"제품명": row['제품명']}) for _, row in med_product_df.iterrows()]
    documents_medi_info = [Document(page_content=combine_pdf(row), metadata={"제목": row['제목']}) for _, row in med_info_df.iterrows()]

    OpenAIembedding = OpenAIEmbeddings()
    SolarEmbedding = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

    embedding_model = OpenAIembedding
    #embedding_model = SolarEmbedding
    
    kiwibm25_faiss_73_sup_product = retriever_sup(documents_sup_product, embedding_model)
    kiwibm25_faiss_73_medi_product = retriever_medi(documents_medi_product, embedding_model)
    kiwibm25_faiss_73_medi_info = retriever_medi(documents_medi_info, embedding_model)


    
    def prompt_classification(llm, prompt):
        if '추천' in prompt and '분석' not in prompt:
            response = '제품 추천을 원하는 문장입니다.'
        elif '분석' in prompt and '추천' not in prompt:
            response = '문제 해결을 위한 분석을 원하는 문장입니다.'
        else:
            messages = [
                SystemMessage(
                    content="You are an assistant designed to classify user requests into one of two categories: (1) product recommendation or (2) problem-solving analysis. Your goal is to accurately determine whether the user is asking for product suggestions or seeking analytical help to resolve a problem based on the content and intent of the user's query."
                ),
                HumanMessage(
                    content=f"{prompt}라는 문장이 제품 추천을 원하는 문장인지 문제 해결을 위한 분석을 원하는 문장인지 아니면 둘 다 아닌지 답변해줘"
                )
                ]
        response = llm.invoke(messages).content
        return response
    
    def make_retrieveQA_form(contexts,question):
        prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + contexts + "Question:" + question + "Helpful Answer: "
        return prompt

    def solar_generate_text(client, prompt, adapter_id, max_new_tokens):
        answer = client.generate(prompt, adapter_id=adapter_id, max_new_tokens=max_new_tokens).generated_text
        return answer

    def qa_chain_generate_text(qa_chain, prompt):
        if qa_chain =='solar_qa_chain':
            answer = solar_generate_text(client, prompt, adapter_id, max_new_tokens)
        else:
            answer = qa_chain.invoke({"query": prompt})['result']
        return answer
    
    def generate_chat(llm, prompt):
        messages = [
            SystemMessage(
                content="You are an helpful assistant."
            ),
            HumanMessage(
                content=prompt
            )
        ]
        response = llm.invoke(prompt).content
        return response

    
    # GraphState 상태를 저장하는 용도로 사용합니다.
    class GraphState(TypedDict):
        question: str  # 질문
        context: str  # 문서의 검색 결과
        answer: str  # 답변
        relevance: str  # 답변의 문서에 대한 관련성
        prompt_type : str # 분석 / 추천
        product_type : str # 영양제 / 약
        med_recommend_list : list

    def product_type(state : GraphState) -> GraphState:
        if state['product_type'] == 'medi':
            return GraphState(product_type = 'medi')
        
    def prompt_class(state : GraphState) -> GraphState:
        p_type = prompt_classification(base_llm, prompt)
        return GraphState(prompt_type = p_type)
    
    def retrieve_medi_documents(question):
        retrieved_docs = ""
        # Question 에 대한 문서 검색을 retriever 로 수행합니다.) ###
        i_retriever = kiwibm25_faiss_73_medi_info
        retrieved_info_docs = i_retriever.invoke(question)
        
        p_retriever = kiwibm25_faiss_73_medi_product
        retrieved_product_docs = p_retriever.invoke(retrieved_info_docs[0].page_content)
        # 검색된 문서를 context 키에 저장합니다.
        retrieved_info, medi_product = "", ""
        for info in retrieved_info_docs:
            retrieved_info += info.page_content + '\n'
        for m_product in retrieved_product_docs:
            medi_product += m_product.page_content + '\n'
        retrieved_docs = retrieved_info + "를 고려하여 제품을 추천하는데, \n " + medi_product + "이 정보들을 참고해서 Question 에 대한 답을 해줘 Question : " + question #### prompt에 들어갈 contents
        return retrieved_docs ###

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    def retrieve_document(state: GraphState) -> GraphState:
        # Question 에 대한 문서 검색을 retriever 로 수행합니다.
        if state['product_type'] == 'medi':
            retrieved_docs = retrieve_medi_documents(state["question"])
        else:
            retriever = kiwibm25_faiss_73_sup_product
            retrieved_docs = retriever.invoke(state["question"]) ###
        # 검색된 문서를 context 키에 저장합니다.
        return GraphState(context = retrieved_docs) ###

    #Solar Chatbot model을 사용하여 답변을 생성합니다.
    def llm_answer(state: GraphState) -> GraphState:
        if state['product_type'] == 'medi':
            return GraphState(
            #answer=solar_generate_text(client, prompt, adapter_id, max_new_tokens),
            answer = generate_chat(base_llm, state['context']),
            context=state["context"],
            question=state["question"],
        )
        else:       
            #context_n = len(query.split(','))+1
            #state['context'] = state['context'][:context_n]
            contexts = ", ".join([state['context'][i].page_content for i in range(len(state['context']))])
            prompt = make_retrieveQA_form(contexts,state['question'])
            return GraphState(
                #answer=solar_generate_text(client, prompt, adapter_id, max_new_tokens),
                answer = qa_chain_generate_text(qa_chain, prompt),
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
    solar_chat_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-1-mini-chat")

    retriever = kiwibm25_faiss_73_sup_product
    
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
    #workflow.add_node("classification", prompt_classification)
    workflow.add_node("product", product_type)
    workflow.add_node("retrieve", retrieve_document)  # 에이전트 노드를 추가합니다.
    workflow.add_node("llm_answer", llm_answer)  # 정보 검색 노드를 추가합니다.
    workflow.add_node("relevance_check", relevance_check)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.

    #workflow.add_edge("classification","retrieve") # 프롬프트 타입 -> 검색
    workflow.add_edge("product", "retrieve")
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

    workflow.set_entry_point("product")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # recursion_limit: 최대 반복 횟수, thread_id: 실행 ID (구분용)
    config = RunnableConfig(recursion_limit=13, configurable={"thread_id": "SELF-RAG"})

    #query = "멀티비타민 올인원, 밀크씨슬"
    #prompt = f"저의 성별은 남성, 나이는 성인 (19세 이상)입니다. 제가 지금 먹고 있는 영양제 제품들은 {query} 이고 이렇게 먹으면 적절한지 권장 섭취량과 현재 복용중인 영양성분을 비교해서 확인해주세요."

    # prompt = "두통이 있는데 어떤 약을 먹는게 좋을까?"
    pb_token = pb(api_token=PREDIBASE_API_KEY)
    client = pb_token.deployments.client("solar-1-mini-chat-240612")
    adapter_id, max_new_tokens ="AIMedicine/1", 2000

    inputs = GraphState(question=prompts, product_type = chat_type)
    # qa_chain = solar_qa_chain  # solar_qa_chain or gpt4o_qa_chain
    qa_chain = gpt4o_qa_chain
    base_llm = gpt4o_llm
    retriever = kiwibm25_faiss_73_sup_product
    output = app.invoke(inputs, config=config)
    tab = 'sup'
    return output['answer']