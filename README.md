# NutriPharmAI
> This project developed a pharmacist chatbot using an LLM and LangChain-managed pipelines to provide accurate recommendations and analyses of supplements and medications, ensuring user safety through interaction checks.
---
Upstage 2024 Global AI week AI Heckathon 출품작 (BisAI)
---
## Project Overview

Our project is a pharmacist chatbot AI that analyzes the ingredients in multiple supplements or medications. It identifies cases where the dosage exceeds recommended levels, potentially causing side effects, or detects harmful ingredient interactions. When necessary, the system also recommends products available in Korea. We have implemented this solution using various Upstage APIs, integrated with LangGraph to handle multiple scenarios effectively.

<img width="1251" alt="image" src="https://github.com/user-attachments/assets/4dba634d-026b-49e2-9447-7348749b108f">

The technology stack used in this project includes Python for development, with frameworks such as LangChain for managing task pipelines, and LLMs for generating responses. 
The Upstage API was utilized in several key areas: 
- it facilitated chatbot interactions to determine pipeline direction and generate final answers
- it provided embeddings for database creation and query retrieval searches
- it was employed for Ground Check to verify the reliability of responses
- it processed image data using OCR
- it handled PDF file processing through document parsing.

<img width="1148" alt="image" src="https://github.com/user-attachments/assets/b1f23602-61c0-48d1-a03b-3d0f77e801ee">

In this work, we primarily used the Gradio, LangChain and the Upstage API for solar-1-mini-chat-240612 model fine-tuning, generation, and checking groundness.

## Installation method
You have to build the conda enviroment. After clone the repository, please make the conda enviroment by Anaconda.

> $ conda create -n 'env_name' python==3.10.14

> $ conda activate 'env_name'

> $ pip install -r requirements.txt

> $ pip install gradio 
- because of module version conflict, please install some module after create conda enviroment! Although, there will be some error, it works well. So don't be worry about it.   (updatad in 24.08.22)

~~> $ conda env create -f enviroment.yml~~

## Usage Instruction
1. If you want to use this tool, you have to need 4 of api keys.
    - Langchain API key
    - OpenAI API key
    - Upstage API key
    - Predibase API key

2. After get the key, please insert it in [Module/RAG.py](https://github.com/sehooni/NutriPharmAI/blob/90400286cd48f97b8cd62862b85e33a75896a766/Module/RAG.py#L42C1-L45C26).
    Then, you can use the RAG function to get the correct answers from db.

3. Furthermore, you also have to **Predibase API key** in [app.py](https://github.com/sehooni/NutriPharmAI/blob/90400286cd48f97b8cd62862b85e33a75896a766/app.py#L10). 

4. Before run the app.py code, you have to fine-tune the model in the Predibase. There are 2 fine-tuned models in the project; Answer_bot and medi_chatbot. After train the model, there will be adapter_id which have to fill in [RAG.py](https://github.com/sehooni/NutriPharmAI/blob/e2098d525c1216dac97e716394fd451d9a612934/Module/RAG.py#L46C1-L49C26) and [app_final.py](https://github.com/sehooni/NutriPharmAI/blob/e2098d525c1216dac97e716394fd451d9a612934/app_final.py#L13)!
    
    4.1 Train the answer_bot
        
    with **./dataset/final_nutrient_prompt_completion.jsonl**, please use the name for the model as 'AIMedicine'. It's for the[RAG.py](https://github.com/sehooni/NutriPharmAI/blob/90400286cd48f97b8cd62862b85e33a75896a766/Module/RAG.py#L214).

    4.2 Train the medi_chatbot

    with **./dataset/dataset_prompt_completion.jsonl**, please use the name for the model as 'medicine_suggest_model'. It's for the [app_final.py](https://github.com/sehooni/NutriPharmAI/blob/e3b2c9ec5ccd9b610808af1ed71812dadef53f24/app.py#L60).

5. In the ends, please **run the app.py** code in your conda enviroments. Then you can access to our webpage which are undisclosed.

## About the Webpage
In the webpage, you can use 2 service, "Check the recommended dosage of nutritional supplements" and "Chat with Chatbot". Just click the one section!

![web_description1](https://github.com/user-attachments/assets/b43f14d4-e107-4cef-89a3-78bf82855d77)

1. **"Check the recommended dosage of nutritional supplements"**

    In this section, you can check the recommended dosage of nutritional supplements.
    
    please input the information; heights, weights, age, sex.

    Additionally, you can choose the nutritional supplements from the nutritional supplement selection. If the nutritional supplements you want to check do not exist in the selection, just write down next to selection and push the add button. 
    Then the prompt will be generated. 
    
    Please write the question what you want. We suggest that there are baseline questions, so you just click the 'generation' button! 

    Then you can get the recommended dosage of nutritional supplements from our service! 

2. **"Chat with Chatbot"**

    In this section, you can chat with our chatbot fine-tuned by the SolarAI.
    
    Suggestion input message:  **"your symptoms"** with **"please suggest the medicine"**. 

During the Global AI week AI Hackaton presentation periods, this webpage will be opened for 72hours!

