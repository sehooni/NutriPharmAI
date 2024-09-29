# Import module
import gradio as gr
# from predibase import Predibase, FinetuningConfig, DeploymentConfig
from langchain.schema import AIMessage, HumanMessage
from openai import OpenAI
from Module import RAG
from PIL import Image
import requests
import numpy as np
import pandas as pd
import faiss

api_key = '...'
client = OpenAI(api_key=api_key)


# define the function
def image_to_text(filename):
    s_api_key = "..."
    url = "https://api.upstage.ai/v1/document-ai/ocr"
    headers = {"Authorization": f"Bearer {s_api_key}"}
    files = {"document": open(filename, "rb")}
    response = requests.post(url, headers=headers, files=files)
    
    query_text = response.json()["text"]
    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        input= query_text,
        model="text-embedding-3-small"
    )
    df = pd.read_csv('./dataset/nutrient_data_final_2.csv')
    query_emb = response.data[0].embedding
    query_emb = np.array([query_emb])
    faiss_index = faiss.read_index('./full_emb.index')
    faiss_index.nprobe = 5
    result = faiss_index.search(query_emb, 10)[1]
    cand_list = []
    for name in df.loc[result[0]]["이름"][:5]:
        cand_list.append(name)
    cand_list.append('없음')
    cand_out = ''
    for i, idx in enumerate(cand_list):
        if i == len(cand_list)-1:
            cand_out += f'{idx}'
        else:
            cand_out += f'{idx}, '
    return cand_out

def update_dropdown(text):
    options = [option.strip() for option in text.split(',')]
    return gr.Dropdown(choices=options, multiselect=True, interactive=True)


## insert function
def insert2pill_list(input):
    return input

def sentence_builder_addition(tall, weight, age, sex, pill, pill_addition, pill_img):
    total_pill = []
    if pill == None :
        pill = []
    if len(pill_addition) == 0:
        pill_addition = []
        total_pill.extend(pill_addition)
    else:
        pill_add =''.join(pill_addition)
        pill_add2 = pill_add.split(',')
        total_pill.extend(pill_add2)
        
    if pill_img == None:
        pill_img = []
    total_pill.extend(pill)
    total_pill.extend(pill_img)
        
    if len(total_pill) != 0:
        pill_str = ', '.join(total_pill)
    
    input_prompt = f"키 : {tall}cm \n\n몸무게 : {weight}kg \n\n나이 및 성별 : {age} {sex} \n\n복용 중인 영양제 : {pill_str}"
    return input_prompt

def sentence_builder_new(tall, weight, age, sex, pill, pill_info):
    total_pill = []
    if pill == None :
        pill = []
        
    if pill_info == None:
        pill_info = []
    total_pill.extend(pill)
    total_pill.extend(pill_info)
        
    if len(total_pill) != 0:
        pill_str = ', '.join(total_pill)
    else:
        pill_str = None
    
    input_prompt = f"키 : {tall}cm \n\n몸무게 : {weight}kg \n\n나이 및 성별 : {age} {sex} \n\n복용 중인 영양제 : {pill_str}"
    return input_prompt

def sentence_builder_add(pill_addition, pill_img):
    total_pill = []
    if len(pill_addition) == 0:
        pill_addition = []
        total_pill.extend(pill_addition)
    else:
        pill_add =''.join(pill_addition)
        pill_add2 = pill_add.split(',')
        total_pill.extend(pill_add2)
        
    if pill_img == None:
        pill_img = []
        
    if '없음' not in pill_img:
        total_pill.extend(pill_img)
        
    if len(total_pill) != 0:
        pill_str = ', '.join(total_pill)
    
    return pill_str

##rag로 주성분 가져오기
def get_pill_prompt(pill, pill_addition):
    pill_a1 =''.join(pill_addition)
    pill_a2 = pill_a1.split(',')
    pill.extend(pill_a2)
    pill_stri = ', '.join(pill)
    pill_prompt = f"복용 중인 영양제 : {pill_stri}"
    return pill_prompt

        
## 영양제 prompt maker
def call_model(sex, age, pill, pill_info, chat):
    total_pill = []
    if pill == None :
        pill = []
        
    if pill_info == None:
        pill_info = []
    total_pill.extend(pill)
    total_pill.extend(pill_info)
        
    if len(total_pill) != 0:
        pill_str = ', '.join(total_pill)
    # lorax_client = pb.deployments.client("solar-1-mini-chat-240612")
    input_prompt = f"저의 성별은 {sex}, 나이는 {age}입니다.\n 제가 지금 먹고 있는 영양제 제품들은 {pill_str}이고,\n {chat}"
    output = RAG.LangGraph(pill_str, input_prompt, 'sup')
    # output = lorax_client.generate(input_prompt, adapter_id="AIMedicine/1", max_new_tokens=10000).generated_text
    return output

def call_model2(chat):
    output = RAG.LangGraph('', chat, 'medi')
    # output = lorax_client.generate(input_prompt, adapter_id="AIMedicine/1", max_new_tokens=10000).generated_text
    return output

def response(message, history): 
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content":"넌 약사야."},
            {"role":"user", "content":f"{message}"}
        ]
    )
    chat_response = completion.choices[0].message.content
    final_response = call_model2(chat_response)
    return final_response

def create_radio_options(text):
    addition_choice = gr.Radio(choices=text, label="해당하는 제품을 선택하여주세요.", info='해당하는 제품이 없으면, 해당없음을 선택하세요.')
    return addition_choice

def insert_Data(text):
    return "다이어트 유산균 비에날씬, 비타앤 히알루론산 피치맛, 고흡수 마그네슘"

# Header & pill list define
_HEADER_ = '''
<h1><b>💊 NutriPharmAI 💊 </b></h1>
<h4> "Nutrition" (영양), "Pharmacy" (약국), 그리고 "AI"를 조합하여, 영양과 약물 관리를 전문적으로 다루는 AI를 표현합니다.</h4>
<h3><b>키, 몸무게, 나이, 성별, 복용 영양제를 입력해 권장 복용량을 추천해줍니다!</b></h3>
<h4><b> 🧑🏻‍💻 Constructed by bisAI (for Upstage 2024 Global AI Week AI Hackaton) </b></h4>
'''
pill_list = ["다이어트 유산균 비에날씬", "비타앤 히알루론산 피치맛", "고흡수 마그네슘", "루테인 지아잔틴 아스타잔틴", "멀티비타민 올인원", "츄어블 L-테아닌 100mg", "프로바이오 생유산균", "밀크씨슬+", "칼슘마그네슘아연비타민D", "엠에스엠파우더 MSM Powder with OptiMSM", "프리미엄 항산화 혈압 건강 코엔자임Q10 코큐텐 플러스", "포 우먼 멀티 비타민 미네랄", "에버콜라겐 타임", "쎌티아이 이눌린", "루테인오메가3", "다이어트 유산균 비에날씬 프로", "락토핏 생유산균 플러스 포스트바이오틱스"]

  

with gr.Blocks(title="pill assistant") as demo:
    gr.Markdown(_HEADER_)
   
    with gr.Tab("영양제 권장 복용량 확인하기"):
        with gr.Row():
            gr.Markdown('''<h2><b> 기본 정보를 설정해주세요.😄 </b></h2>''')
            
        with gr.Row():
            with gr.Column(scale=1):
                tall = gr.Textbox(label="키(cm)", placeholder='당신의 키(신장)를 넣어주세요!')
                weight = gr.Textbox(label="몸무게(kg)", placeholder='당신의 몸무게를 넣어주세요!')
            with gr.Column(scale=1):
                age = gr.Radio(["어린이 (3~8세)", "청소년 (9~18세)", "성인 (19세 이상)", "노인 (65세 이상)"], label="나이", info='당신의 나이를 선택해주세요!')
                sex = gr.Radio(["남성", "여성"], label="성별", info='당신의 성별을 선택해주세요!')

        with gr.Row():
            gr.Markdown('''<h2><b> 복용 중인 영양제를 설정해주세요.😄 </b></h2>''')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('''<h3><b> 해당하는 영양제를 선택해주세요.😄 </b></h3>''')
                pill = gr.Dropdown(
                    pill_list, multiselect=True, label="영양제", info="복용 중인 영양제를 선택하여 주세요!"
                )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('''<h3><b> 해당하는 영양제가 없다면? </b></h3>''')
                addi_pill = gr.Textbox(visible=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('''<h4><b> 직접 추가하기 </b></h4>''')
                        with gr.Row():
                            pill_input = gr.Textbox(label='영양제', info='해당하는 영양제가 없다면 입력하여 주세요!', placeholder= '영양제 이름을 입력해주세요!')
                            pill_input_bt = gr.Button("추가하기")
                            pill_tmp = gr.Textbox(visible=False)
                            pill_input_bt.click(insert2pill_list, inputs=[pill_input], outputs=pill_tmp)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('''<h4><b> 이미지로 추가하기 </b></h4>''')
                        with gr.Row():
                            img = gr.Image(type="filepath", height=300, width=400)
                            img_candidate = gr.Textbox(visible=False)
                            img.change(image_to_text, inputs=img, outputs=img_candidate)
                        img_button = gr.Button("check")
                        candidate_list = gr.Dropdown(choices=[], multiselect=True, label='해당 영양제 선택', interactive=True)
                        img_button.click(fn=update_dropdown, inputs=[img_candidate], outputs=[candidate_list])
                        
                pill_tmp.change(sentence_builder_add, inputs=[pill_tmp, candidate_list], outputs=addi_pill)
                candidate_list.change(sentence_builder_add, inputs=[pill_tmp, candidate_list], outputs=addi_pill)      
                  
            with gr.Column(scale=1):
                gr.Markdown('''<h4><b> 데이터 확인</b></h4>''')
                selection_tmp = gr.Textbox(visible=False)
                """ 여기에 데이터 없는거 검색해서 연결하도록 설정 """
                addi_pill.change(insert_Data, inputs=[addi_pill], outputs=selection_tmp)
                selection_list_button = gr.Button('search')
                selection_list = gr.Dropdown(choices=[], multiselect=True, label='타당성 확인', interactive=True)
                # 드롭다운에서 선택한거를 바탕으로 datacheckbox 수정 후 보여주기
                selection_list_button.click(fn=update_dropdown, inputs=[selection_tmp], outputs=[selection_list])
                
            
        with gr.Row():
            with gr.Column():
                gr.Markdown('''<h5>기본 설정 및 복용 중인 영양제 확인 💊💊</h5>''')
                check_out = gr.Textbox(label='확인 창', visible=True)


            pill.change(sentence_builder_new, inputs=[tall, weight, age, sex, pill, selection_list], outputs=check_out)
            selection_list.change(sentence_builder_new, inputs=[tall, weight, age, sex, pill, selection_list], outputs=check_out)
            
        with gr.Row():    
            with gr.Column():
                gr.Markdown('''<h2><b> 질문을 입력하여 주세요.😄 </b></h2>''')
                chat = gr.Textbox(label="Chat", placeholder='제품들을(를) 같이 먹고 있는데 적합하게 먹고 있어? 권장섭취량을 참고해서 지금 먹고 있는 영양제가 권장량에 적합한지 비교해줘', value='제품들을(를) 같이 먹고 있는데 적합하게 먹고 있어? 권장섭취량을 참고해서 지금 먹고 있는 영양제가 권장량에 적합한지 비교해줘')

                submit = gr.Button("생성하기", size='sm')

        with gr.Row():
            output = gr.Textbox(label="권장 복용 설명")
            
        submit.click(fn=call_model, inputs=[sex, age, pill, selection_list, chat], outputs=[output])
    
    
    
    with gr.Tab("채팅하기"):
        gr.ChatInterface(
            fn=response,
            title="Pill assistant AI with Chatbot 🤖",
            description="챗봇과의 채팅을 통해 원하는 대화를 나눠보세요!",
            theme="soft",
            chatbot=gr.Chatbot(height=750)             
        )
demo.launch(share=True)