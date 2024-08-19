# NutriPharmAI
> "Nutrition" (영양), "Pharmacy" (약국), 그리고 "AI"를 조합하여, 영양과 약물 관리를 전문적으로 다루는 AI를 표현합니다.
---
Upstage 2024 Global AI week AI Heckaton 출품작 (BisAI)
---
## Project Overview
We have developed a model that analyzes the nutritional content of entered supplements to ensure they meet daily recommended intakes, and a pharmacist chatbot that recommends medication based on reported symptoms.


Our Chat model analyzes the names of supplements and medications entered by users, identifies overlapping ingredients, and provides safe dosage recommendations.

![Overview1](https://github.com/user-attachments/assets/35d3427e-5b1a-4bc4-9f8a-6404ef174471)

We have fine-tuned an LLM model to better understand user questions and provide specialized responses, and by using RAG (Retrieval-Augmented Generation), we can analyze complex product ingredient and nutritional data.

![Overview2](https://github.com/user-attachments/assets/211a647b-a6c9-4d5d-ad28-9a608ec9bfd5)

In this work, we primarily used the Gradio, LangChain and the Upstage API for solar-1-mini-chat-240612 model fine-tuning, generation, and checking groundness.

## Installation method
You have to build the conda enviroment. After clone the repository, please make the conda enviroment by Anaconda.

> $ conda env create -f enviroment.yml

## Usage Instruction
1. If you want to use this tool, you have to need 4 of api keys.
    - Langchain API key
    - OpenAI API key
    - Upstage API key
    - Predibase API key

2. After get the key, please insert it in [Module/RAG.py](https://github.com/sehooni/NutriPharmAI/blob/90400286cd48f97b8cd62862b85e33a75896a766/Module/RAG.py#L42C1-L45C26).
    Then, you can use the RAG function to get the correct answers from db.

3. Furthermore, you also have to **Predibase API key** in [app.py](https://github.com/sehooni/NutriPharmAI/blob/90400286cd48f97b8cd62862b85e33a75896a766/app.py#L10). 

4. In the ends, please run the app.py code in your conda enviroments. Then you can access to our webpage which are undisclosed.

During the Global AI week AI Hackaton presentation periods, this webpage will be opened for 72hours!

