import os
import cv2
import logging
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from dataclasses import dataclass
from multiprocessing import Value

# import braceexpand

import torch
import torchvision.datasets as datasets
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
import glob
import re

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

Templates = ['CLASSNAME.',
            'a photomicrograph showing CLASSNAME.',
            'a photomicrograph of CLASSNAME.',
            'an image of CLASSNAME.',
            'an image showing CLASSNAME.',
            'an example of CLASSNAME.',
            'CLASSNAME is shown.',
            'this is CLASSNAME.',
            'there is CLASSNAME.',
            'a histopathological image showing CLASSNAME.',
            'a histopathological image of CLASSNAME.',
            'a histopathological photograph of CLASSNAME.',
            'a histopathological photograph showing CLASSNAME.',
            'shows CLASSNAME.',
            'presence of CLASSNAME.',
            'CLASSNAME is present.',
            'an H&E stained image of CLASSNAME.',
            'an H&E stained image showing CLASSNAME.',
            'an H&E image showing CLASSNAME.',
            'an H&E image of CLASSNAME.',
            'CLASSNAME, H&E stain.',
            'CLASSNAME, H&E.'
            ]


class PKDataset(Dataset):
    def __init__(self, dataset, transform=None, istrain = True):
        self.dataset = dataset
        self.transform = transform
        self.istrain = istrain

    def __len__(self):
        return len(self.dataset)
    
    def add_template(self, entity_text):
        
        if random.random()< 0:
            return entity_text
        rand_temp = Templates[random.randint(0,len(Templates)-1)]
        template_text = rand_temp.replace('CLASSNAME',entity_text)
        
        return template_text

    def __getitem__(self, index):
        text, did, tid, attr = self.dataset[index]
        
        # if self.istrain and (attr.startswith('main') or attr.startswith('syn')):
        #     text = self.add_template(text)

        if self.transform is not None:
            text = self.transform(text)

        return text, did, tid, attr

## 
class PathKnowledge(Dataset):
    """Pathology Knowledge.

    Dataset statistics:
        - entities: 4718.
        - instances: 41096 (train)
                     679, 5441 (query,test) for synonyms
                     950, 1948 (query,test) for definitions
                     774, 774  (query,test) for histologic features
                     251, 251  (query,test) for cytologic features
    """

    def __init__(self, dataset_root):
        train_dir = os.path.join(dataset_root, 'PathTissueAttr_train.csv')
        # query_dir = os.path.join(dataset_root, 'PathKnowledge_syn_query.csv')
        test_syn_dir = os.path.join(dataset_root, 'PathKnowledge_syn_test.csv')
        test_def_dir = os.path.join(dataset_root, 'PathKnowledge_def_test.csv')
        test_his_dir = os.path.join(dataset_root, 'PathKnowledge_his_test.csv')
        test_cyt_dir = os.path.join(dataset_root, 'PathKnowledge_cyt_test.csv')
        test_tempsyn_dir = os.path.join(dataset_root, 'PathKnowledge_tempsyn_test.csv')
        test_pathout_dir = os.path.join(dataset_root, 'PathOut_reid.csv')
        self.train_list = pd.read_csv(train_dir,sep='\t',).values.tolist()
        # self.query_list = pd.read_csv(query_dir,sep='\t').values.tolist()

        self.test_dict = {}
        self.test_dict['syn'] = pd.read_csv(test_syn_dir,sep='\t').values.tolist()
        self.test_dict['def'] = pd.read_csv(test_def_dir,sep='\t').values.tolist()
        self.test_dict['his'] = pd.read_csv(test_his_dir,sep='\t').values.tolist()
        self.test_dict['cyt'] = pd.read_csv(test_cyt_dir,sep='\t').values.tolist()
        self.test_dict['tempsyn'] = pd.read_csv(test_tempsyn_dir,sep='\t').values.tolist()
        self.test_dict['pathout'] = pd.read_csv(test_pathout_dir,sep='\t').values.tolist()

        self.train = self.process_list(self.train_list)
        self.test = self.process_list(self.test_dict)

        self.query,self.gallery = self.get_query_gallery(self.test_dict)

    def format_data(self, data_list):
        did_container = set()
        for item in data_list:
            entity_name,entity_text = item[0],item[1]
            did = int(entity_name.split('_')[0])
            if did not in did_container:
                did_container.add(did)
        did2label = {did: label for label, did in enumerate(did_container)}

        dataset = []
        for item in data_list:
            entity_name,entity_text = item[0],item[1]
            did = int(entity_name.split('_')[0])
            tid = int(entity_name.split('_')[1].split('t')[1])
            attr = entity_name.split('_')[2]
            dataset.append((entity_text, did2label[did], tid, attr))
        return dataset

    def process_list(self, data_list):
        if isinstance(data_list,dict):
            dataset = {}
            for k,v in data_list.items():
                dataset[k] = self.format_data(v)
        elif isinstance(data_list,list):
            dataset = self.format_data(data_list)

        return dataset
    
    def get_query_gallery(self, data_list):
        if isinstance(data_list,dict):
            query = {}
            gallery = {}
            for k,v in data_list.items():
                query[k] = []
                gallery[k] = []
                for instance in v:
                    entity_name,entity_text = instance[0],instance[1]
                    did = int(entity_name.split('_')[0])
                    tid = int(entity_name.split('_')[1].split('t')[1])
                    attr = entity_name.split('_')[2]
                    if attr.startswith('main'):
                        tid += 500  # to avoid tissue filter during reid evaluation
                        query[k].append((entity_text, did, tid, attr))
                    else:
                        gallery[k].append((entity_text, did, tid, attr))
        elif isinstance(data_list,list):
            query = []
            gallery = []
            for instance in data_list:
                entity_name,entity_text = instance[0],instance[1]
                did = int(entity_name.split('_')[0])
                tid = int(entity_name.split('_')[1].split('t')[1])
                attr = entity_name.split('_')[2]
                if attr.startswith('main'):
                    tid += 500  # to avoid tissue filter during reid evaluation
                    query.append((entity_text, did, tid, attr))
                else:
                    gallery.append((entity_text, did, tid. attr))

        return query, gallery


class UMLS_Dataset(Dataset):
    def __init__(self,mrdef_csv_file, umls_kg_file, umls_cui_file):
        self.mrdef_info = pd.read_csv(mrdef_csv_file)
        self.mrdef_cui_list = self.mrdef_info.iloc[:,0]
        self.mrdef_name_list = self.mrdef_info.iloc[:,1]
        self.mrdef_def_list = self.mrdef_info.iloc[:,2]

        self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_kg_source_list = self.umls_kg_info.iloc[:,0]
        self.umls_kg_target_list = self.umls_kg_info.iloc[:,1]
        self.umls_kg_edge_list = self.umls_kg_info.iloc[:,2]

        self.umls_cui_info = pd.read_csv(umls_cui_file)
        self.umls_cui_source_list = self.umls_cui_info.iloc[:,0]
        self.umls_cui_target_list = self.umls_cui_info.iloc[:,1]

        self.umls_data_len = len(self.umls_kg_info)
        self.mrdef_data_len = len(self.mrdef_info)
        print('UMLS data length: ',self.umls_data_len)
        print('MRDEF data length: ',self.mrdef_data_len)
        self.select_umls_ratio = self.umls_data_len/(self.umls_data_len+self.mrdef_data_len)
    
    def __len__(self):
        return int(self.umls_data_len+self.mrdef_data_len)
    
    def __getitem__(self, idx):
        if random.random() < self.select_umls_ratio:
            select_idx = random.randint(0,self.umls_data_len-1)
            text_h = self.umls_kg_source_list[select_idx]
            cui_h = self.umls_cui_source_list[select_idx]
            text_t = self.umls_kg_target_list[select_idx]
            cui_t = self.umls_cui_target_list[select_idx]
            text_r = self.umls_kg_edge_list[select_idx]
            if random.random()<0.5:
                input_text = text_h + ' [SEP] ' + text_r
                pos_text =  text_t
                cui = cui_t
            else:
                input_text = text_r + ' [SEP] ' + text_t
                pos_text =  text_h
                cui = cui_h
        else:
            select_idx = random.randint(0,self.mrdef_data_len-1)
            input_text = self.mrdef_name_list[select_idx]
            pos_text = self.mrdef_def_list[select_idx]
            cui = self.mrdef_cui_list[select_idx]
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        try: 
            if cui[0] == 'C':
                sample['cui'] = cui
            else:
                sample['cui'] = str(0)
        except:
            sample['cui'] = str(0)
        return sample
        
class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
