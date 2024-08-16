import gradio as gr
from predibase import Predibase, FinetuningConfig, DeploymentConfig

pb = Predibase(api_token="...")

# def sentence_builder(sex, age, pill, component, quest):
#     input_prompt = f"저의 성별은 {sex}, 나이는 {age}입니다.\n 제가 지금 먹고 있는 영양제 제품들은 {pill}이고,\n각 영양제의 성분은 {component}입니다.\n {quest}"
#     return input_prompt

chat = '이렇게 영양제를 복용한다면 어떤 성분이 일일 상한섭취량을 초과하나요?'
comp = '비타민B1: 0.6, 비타민B2: 0.7, 나이아신 (비타민B3): 7.5, 판토텐산 (비타민B5): 2.5, 비타민B6: 0.75, 비오틴 (비타민B7): 15.0, 엽산 (비타민B9): 200.0, 아연: 4.25, 셀레늄: 27.5, 크롬: 15.0, 코엔자임Q10: 100.0 '
        

def call_model(sex, age, pill):
    lorax_client = pb.deployments.client("solar-1-mini-chat-240612")
    input_prompt = f"저의 성별은 {sex}, 나이는 {age}입니다.\n 제가 지금 먹고 있는 영양제 제품들은 {pill}이고,\n각 영양제의 성분은 {comp}입니다.\n {chat}"
    output = lorax_client.generate(input_prompt, adapter_id="medicine_suggest_model/1", max_new_tokens=100).generated_text
    return output

_HEADER_ = '''
<h2><b>💊 Pill_assistant: 약물 보조 AI 💊 </b></h2><h2>
키, 몸무게, 성별, 복용 약물을 입력해 권장 복용량을 추천해줍니다!

영양제 or 의약품 tab을 선택하여 주세요!
'''

with gr.Blocks(title="pill assistant") as demo:
    gr.Markdown(_HEADER_)
    
    # with gr.Tab("영양제"):
    with gr.Column():
        tall = gr.Textbox(label="키(신장)", placeholder='당신의 키(신장)를 넣어주세요!')
        weight = gr.Textbox(label="몸무게", placeholder='당신의 몸무게를 넣어주세요!')
        sex = gr.Radio(["남성", "여성"], label="성별", info='당신의 성별을 선택해주세요!')
        age = gr.Radio(["어린이 (3~8세)", "청소년 (9~18세)", "성인 (19세 이상)", "노인 (65세 이상)"], label="나이", info='당신의 나이를 선택해주세요!')
        pill = gr.Dropdown(
            ["락포핏", "얼라이브종합비타민", "밀크시슬"], multiselect=True, label="영양제", info="복용 중인 영양제를 선택하여 주세요!"
        )
        # chat = gr.Textbox(label="Chat", placeholder='이렇게 영양제를 복용한다면 어떤 성분이 일일 상한섭취량을 초과하나요?', inputs='이렇게 영양제를 복용한다면 어떤 성분이 일일 상한섭취량을 초과하나요?')
        # chat = '이렇게 영양제를 복용한다면 어떤 성분이 일일 상한섭취량을 초과하나요?'
        # comp = '비타민B1: 0.6, 비타민B2: 0.7, 나이아신 (비타민B3): 7.5, 판토텐산 (비타민B5): 2.5, 비타민B6: 0.75, 비오틴 (비타민B7): 15.0, 엽산 (비타민B9): 200.0, 아연: 4.25, 셀레늄: 27.5, 크롬: 15.0, 코엔자임Q10: 100.0 '
        # # prompt = sentence_builder()
        submit = gr.Button("생성하기")

    with gr.Row():
        output = gr.Textbox(label="권장 복용 설명")

    # with gr.Tab("의약품"):
    #     with gr.Column():
    #         m_fill = gr.Dropdown(
    #             ["타이레놀", "판콜", "스트렙실"], multiselect=True, label="약품",
    #             info="복용 중인 약품을 선택하여 주세요!"
    #         )

    #         m_submit = gr.Button("생성하기")

    #     with gr.Row():
    #         m_output = gr.Textbox(label="권장 복용 설명")

        # quest_item_img = gr.Image(label='퀘스트 아이템', elem_id='quest_item')

    submit.click(fn=call_model, inputs=[sex, age, pill], outputs=[output])
    # submit.click(fn=sentence_builder, inputs=[sex, age, pill, comp, prompt]],)

demo.launch(share=True)