import json
import pandas as pd
import numpy as np

np.random.seed(666)

with open('./utils/templates.json') as f:
    templates = json.load(f)
    templates = templates['templates']

syn_dir = './datasets/knowledge_encoder_dataset/PathKnowledge_syn_test.csv'
syn_info = pd.read_csv(syn_dir,sep='\t').values.tolist()

rand_seq = np.random.choice(len(templates),len(syn_info),replace=True)

temp_syn = []
for k, each_syn in enumerate(syn_info):
    rand_temp = templates[rand_seq[k]]
    new_syn = rand_temp.replace('CLASSNAME', each_syn[1])
    temp_syn.append([each_syn[0],new_syn])

data_pd= pd.DataFrame({'instance_name':[item[0] for item in temp_syn],
                        'text':[item[1] for item in temp_syn]})
data_pd.to_csv("./datasets/knowledge_encoder_dataset/PathKnowledge_tempsyn_test.csv",index=False,sep='\t')

test = 1

