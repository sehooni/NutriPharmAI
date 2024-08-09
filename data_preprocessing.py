import pandas as pd
import os
import json
import jsonlines
from collections import OrderedDict


def revise_data(data):
    data = data.str.replace("\xa0", " ")
    data = data.str.replace("\n\n", "\n")
    return data

def get_jsonl_format(data1, data2, data3, path):
    train_data = OrderedDict()
    message_list = list()
    for i in range(len(data1)):
        for j, question in enumerate(data2):
            prompt = f'{data1[i]}{question}'
            completion = f'{data3[j][i]}'
            if completion == 'nan':
                completion = '없습니다.'
            message_list.append({"prompt": prompt, "completion": completion})
            
    with open(path, "a") as f:
        for data in message_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
          

def main():
    # load the data
    data = pd.read_csv('./dataset/new_medi.csv')

    # get column data
    pill_name = data['제품명']
    composition = data['주성분']
    efficiency = data['이 약의 효능은 무엇입니까?']
    caution_eat = data['이 약을 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?']
    caution_tmp = data['이 약의 사용상 주의사항은 무엇입니까?']
    need_to_know = data['이 약을 사용하기 전에 반드시 알아야 할 내용은 무엇입니까?']
    reaction = data['이 약은 어떤 이상반응이 나타날 수 있습니까?']

    caution_t = {'이 약의 주의사항은 무엇입니까?':[]}
    for i in range(len(need_to_know)):
        if (str(need_to_know[i])=='nan') :
            NTN = ''
        else : 
            NTN = str(need_to_know[i]).strip() + '\n'
            
        if (str(caution_tmp[i]) == 'nan'):
            Cau = ''
        else : 
            Cau = str(caution_tmp[i]).strip()
            
        answer = NTN + Cau
        caution_t['이 약의 주의사항은 무엇입니까?'].append(answer)
    caution_df = pd.DataFrame(caution_t)
    caution = caution_df['이 약의 주의사항은 무엇입니까?']

    question_list = ['의 주성분은 어떻게 됩니까?', ' 의 효능은 무엇입니까?', '을(를) 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?', '의 사용상 주의사항은 무엇입니까?', '은(는) 어떤 이상반응이 나타날 수 있습니까?']
    key_list_tmp = [composition, efficiency, caution_eat, caution, reaction]

    key_list = []
    for key in key_list_tmp:
        data = revise_data(key)
        key_list.append(data)
        
    get_jsonl_format(pill_name, question_list, key_list, './dataset/dataset.jsonl')
    
if __name__ == "__main__":
    main()