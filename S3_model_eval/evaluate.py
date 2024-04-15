import os
import pandas as pd
import numpy as np
import clip
import torch
from metrics import eval_metrics, retrieval_metrics
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from data.dataset import *

from path_model.model import get_cast_dtype, KEP, convert_to_custom_text_state_dict
from data.transforms import image_transform
from collections import OrderedDict
import json
from path_model.ctran import ctranspath
from path_model.factory import load_state_dict
import open_clip
from open_clip.big_vision import load_big_vision_weights
from pathlib import Path
from torch import nn


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Evaluater():
    def __init__(self, model_params, batch_size=256, num_workers=4, seed=666) -> None:
        self.model_type = model_params['model_type']  
        self.device = model_params['device']
        self.max_token = model_params['max_token'] if 'max_token' in model_params else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.model_name = model_params['model_name']

    def biomed_load_checkpoint(self, model, checkpoint_path, strict=True):
        if Path(checkpoint_path).suffix in ('.npz', '.npy'):
            load_big_vision_weights(model, checkpoint_path)
            return {}

        state_dict = load_state_dict(checkpoint_path)
        # detect old format and make compatible with new format
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        # Certain text transformers no longer expect position_ids after transformers==4.31
        position_id_key = 'text.transformer.embeddings.position_ids'
        if position_id_key in state_dict and not hasattr(model, position_id_key):
            del state_dict[position_id_key]
        open_clip.model.resize_pos_embed(state_dict, model)
        open_clip.model.resize_text_pos_embed(state_dict, model)
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        return incompatible_keys
        
    def load_vit_bert_model(self, arch_name, model_path, bert_path, visual_head, logit_scale):
        arch_name = arch_name.replace('/','-')
        if arch_name.lower() =='vit-b-32':
            vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32}
        elif arch_name.lower() =='vit-b-16':
            vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 16}
        else:
            print('No parameters for '+ arch_name)
        cast_dtype = get_cast_dtype('amp')
        image_encoder = 'vit'
        if 'ctp' in self.model_name:
            image_encoder = 'ctp'
        model = KEP(embed_dim=512,
                    vision_cfg= vision_cfg,
                    image_encoder=image_encoder,
                    bert_pretrain= bert_path, 
                    cast_dtype=cast_dtype, 
                    visual_embedding_head=visual_head,
                    logit_scale = logit_scale
                    )
       
        if '16' in self.model_name:
            model_root = '../pretrained_model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/'
            with open(model_root + 'open_clip_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_cfg = config['model_cfg']
            checkpoint_path = model_root + 'open_clip_pytorch_model.bin'
            biomed_model = open_clip.model.CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
            self.biomed_load_checkpoint(biomed_model, checkpoint_path)
            model.visual = biomed_model.visual
        
        elif 'ctp' in self.model_name:
            model_root = '../pretrained_model/CTransPath/ctranspath.pth'
            ctp_model = ctranspath()
            ctp_model.head = nn.Identity()
            state_dict = torch.load(model_root, map_location="cpu")
            missing_keys, unexpected_keys = ctp_model.load_state_dict(state_dict['model'], strict=False)
            print('missing keys: ', missing_keys)
            print('unexpected keys: ', unexpected_keys)

            model.visual = ctp_model
            model.visual.image_size = 224
            model.visual.image_mean = (0.485, 0.456, 0.406)
            model.visual.image_std = (0.229, 0.224, 0.225)
            print('Load pretrained vision encoder success from CTransPath.')
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict,strict=True)
        model.to(self.device)
        model.eval()

        processor = {}
        processor['tokenizer'] = AutoTokenizer.from_pretrained(bert_path,do_lower_case=True, local_files_only=True)
        img_mean = getattr(model.visual, 'image_mean', None)
        img_std = getattr(model.visual, 'image_std', None)
        processor['imgprocessor'] = image_transform(model.visual.image_size,is_train=False,mean=img_mean,std=img_std)

        print('loading vit_bert_model '+ model_path + ' done!')

        self.model = model
        self.processor = processor
        return model,processor


    def load_model(self, model_params, logit_scale=0.04):
        arch_name = model_params['arch_name']
        model_path = model_params['model_path']

        bert_path = model_params['bert_path']
        visual_head = model_params['visual_head']
        return self.load_vit_bert_model(arch_name, model_path, bert_path, visual_head,logit_scale)

    def encode_image_text(self, img_dir_list, caption_list):

        img_dataset = CLIPImageDataset(img_dir_list, self.processor)
        dataloader = DataLoader(img_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        image_embeddings = []
        
        total = len(img_dir_list) // self.batch_size
        pbar = tqdm(total=total, position=0)
        with torch.no_grad():
            for images in dataloader:
                images = images.to(self.device)
                img_features = self.model.encode_image(images).detach().cpu().numpy()
                image_embeddings.extend(img_features)
                pbar.update(1)
            pbar.close()

        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        
        cap_dataset = CLIPCaptioningDataset(caption_list)
        dataloader = DataLoader(cap_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        text_embeddings = []
        total = len(caption_list) // self.batch_size

        pbar = tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:

                try:
                    text_inputs = self.get_tokenizer(captions,self.processor['tokenizer'],max_length=self.max_token).to(self.device)
                except:
                    text_inputs = clip.tokenize(captions, truncate=True).to(self.device)
                
                if self.model_type == 'vit_bert':
                    text_out = self.model.encode_text(text_inputs)
                elif self.model_type == 'biomed_bert':
                    text_inputs = self.processor['tokenizer'](captions, context_length=256).to(self.device)
                    text_out = self.model.encode_text(text_inputs)
                else:
                    text_out = self.model.encode_text(text_inputs)
                
                text_embeddings.extend(text_out.detach().cpu().numpy())
                pbar.update(1)
            pbar.close()

        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        return image_embeddings, text_embeddings
    

    def get_dataset_info(self, data_root, image_root, dataset_name, dataset_type, image_key, caption_key, label_key = None, unique = False, sep = '\t'):
        
        dataset = pd.read_csv(os.path.join(data_root, dataset_name, dataset_name + '_' + dataset_type + '.csv'), sep=sep)
        
        # dataset = dataset.sample(n=int(0.7*dataset.shape[0]),random_state = None)
        
        dataset_info = {}
        print('Loading ' + dataset_name + ' dataset...')
        if dataset_name == 'Kather':
            dataset_info['img_dir_list'] = [os.path.join(image_root, item.split('-')[0],item) for item in dataset[image_key].tolist()]
        else:
            dataset_info['img_dir_list'] = [os.path.join(image_root, item) for item in dataset[image_key].tolist()]
        try:
            if unique:
                dataset_info['caption_list'] = dataset[caption_key].unique().tolist()
            else:
                dataset_info['caption_list'] = dataset[caption_key].tolist()
        except:
            pass

        if label_key is not None:
            dataset_info['unique_labels'] = dataset[label_key].unique().tolist()
            dataset_info['gt_labels'] = dataset[label_key].tolist()

        return dataset_info

    def label2cap(self, data_root, dataset_name):
        with open(data_root + dataset_name+'/' + dataset_name + '_prompts_100.json') as f:
            prompts = json.load(f)
        label_captions = dict()
        types = list(prompts['0']['classnames'].keys())
        for type_name in types:
            label_captions[type_name] = []
            for i in range(0,len(prompts)):
                cls_name = prompts[str(i)]['classnames'][type_name]
                template = prompts[str(i)]['templates']
                cap = template.replace('CLASSNAME',cls_name)
                label_captions[type_name].append(cap)

        return label_captions, len(prompts)

    def zeroshot_eval(
            self, 
            data_root,
            dataset_name,
            dataset_type, 
            image_root, 
            image_key, 
            caption_key, 
            label_key,
            sep = '\t',
            ):

        dataset_info = self.get_dataset_info(data_root, image_root, dataset_name, dataset_type, image_key, caption_key, label_key, unique = False, sep = sep)        
        
        img_dataset = CLIPImageDataset(dataset_info['img_dir_list'], self.processor)
        dataloader = DataLoader(img_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        image_embeddings = []

        print('Image and text embedding...')
        with torch.no_grad():
            self.model.eval()  

            total = len(dataset_info['img_dir_list']) // self.batch_size
            pbar = tqdm(total=total, position=0)
            image_embeddings = []
            for i, batch in enumerate(dataloader):
                images = batch
                images = images.to(device=self.device)
                img_features = self.model.encode_image(images).detach().cpu().numpy()

                image_embeddings.extend(img_features)
                pbar.update(1)
            pbar.close()
            
            template_captions, num_prompts = self.label2cap(data_root, dataset_name)
            
            cap_embeddings = OrderedDict()
            cap_knowledge_embeddings = dict()
            pbar = tqdm(total=len(template_captions), position=0)
            for type_name,caps in template_captions.items():
                if self.model_type == 'vit_bert':
                    text_input = self.get_tokenizer(caps,self.processor['tokenizer'],max_length=self.max_token).to(self.device)
                    text_out = self.model.encode_text(text_input)
                    knowledge_out = self.model.encode_knowledge(text_input) 
                elif self.model_type == 'biomed_bert':
                    text_input = self.processor['tokenizer'](caps, context_length=256).to(self.device)
                    text_out = self.model.encode_text(text_input)
                
                cap_knowledge_embeddings[type_name] = knowledge_out.detach().cpu().numpy()
                cap_embeddings[type_name] = text_out.detach().cpu().numpy()
                pbar.update(1)
            pbar.close()

        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        val_cls = []
        val_sim = []
        cls_embeddings = OrderedDict()
        round_enbeddings = OrderedDict()
        for i in range(num_prompts):
            # print(i)
            cap_labels = []
            each_round = []
            each_know = []
            for type_name,caps in cap_embeddings.items():
                text_embeddings = cap_embeddings[type_name][i,:]
                know_embeddings = cap_knowledge_embeddings[type_name][i,:]
                each_know.append(know_embeddings)
                cap_labels.append(type_name)
                each_round.append(text_embeddings)
                
                if type_name not in cls_embeddings:
                    cls_embeddings[type_name] = [text_embeddings]
                else:
                    cls_embeddings[type_name].append(text_embeddings)
            
            each_round = np.array(each_round)
            each_round = each_round / np.linalg.norm(each_round, axis=1, keepdims=True)
            round_enbeddings[str(i)] = each_round
            
            score = image_embeddings.dot(each_round.T)
            predictions = [cap_labels[np.argmax(i)] for i in score]

            cls_round = eval_metrics(dataset_info['gt_labels'], predictions)
            val_cls.append(cls_round['WF1'])
            
            each_know = np.array(each_know)
            each_know = each_know / np.linalg.norm(each_know, axis=1, keepdims=True)
            sim_score = np.diag(each_know.dot(each_round.T))
            sim_score = sim_score.mean()
            val_sim.append(sim_score)
        
        val_cls = np.array(val_cls)
        
        best_scores = []
        index_labels = dict()
        for ix, (type_name,caps) in enumerate(cap_embeddings.items()):
            index_labels[type_name] = ix
        
        num_labels = []
        for jx, ib in enumerate(image_embeddings):
            num_labels.append(index_labels[dataset_info['gt_labels'][jx]])
            arr = ib.dot(each_round.T)
            best = arr.argsort()[-50:][::-1]
            best_scores.append(best)

        ##
        print('Evaluating...')
        res = np.percentile(val_cls, (25, 50, 75), interpolation='midpoint')
        print('median wF1 (Q1, Q3) is %.3f (%.3f, %.3f)'%(res[1],res[0],res[2]))

    def text2img_retrieval_eval(
            self, 
            data_root,
            dataset_name,
            dataset_type, 
            image_root, 
            image_key, 
            caption_key
            ):
        
        dataset_info = self.get_dataset_info(data_root, image_root, dataset_name, dataset_type, image_key, caption_key)

        ## normalized embedding
        print('Image and text embedding...')
        img_embeddings, text_embeddings = self.encode_image_text(dataset_info['img_dir_list'], dataset_info['caption_list']) 

        ## retrievaling...
        print('Retrievaling...')
        best_scores = []
        for tb in text_embeddings:
            arr = tb.dot(img_embeddings.T)
            best = arr.argsort()[-50:][::-1]
            best_scores.append(best)

        ##
        print('Evaluating...')
        targets = list(range(0, len(img_embeddings)))
        test_metrics = retrieval_metrics(targets, best_scores)
        print('p@10: %.3f'%(test_metrics['p@10']))
        print('p@50: %.3f'%(test_metrics['p@50']))
    
    def img2text_retrieval_eval(
            self, 
            data_root,
            dataset_name,
            dataset_type, 
            image_root, 
            image_key, 
            caption_key, 
            ):
        
        dataset_info = self.get_dataset_info(data_root, image_root, dataset_name, dataset_type, image_key, caption_key)

        ## normalized embedding
        print('Image and text embedding...')
        img_embeddings, text_embeddings = self.encode_image_text(dataset_info['img_dir_list'], dataset_info['caption_list']) 

        ## retrievaling...
        print('Retrievaling...')
        best_scores = []
        
        for ib in img_embeddings:
            arr = ib.dot(text_embeddings.T)
            best = arr.argsort()[-50:][::-1]
            best_scores.append(best)

        ##
        print('Evaluating...')
        targets = list(range(0, len(text_embeddings)))
        test_metrics = retrieval_metrics(targets, best_scores)
        print('p@10: %.3f'%(test_metrics['p@10']))
        print('p@50: %.3f'%(test_metrics['p@50']))

    def get_tokenizer(self,text,tokenizer,max_length):
        token_list = tokenizer(list(text),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt')
        return token_list
        
