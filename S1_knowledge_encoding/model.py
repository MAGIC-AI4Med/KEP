from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig, AutoTokenizer
from utils.transformer import LayerNormFp32, LayerNorm, QuickGELU, TextTransformer

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text

class CLP_BERT(nn.Module):
    def __init__(self,
                bert_model_name: str,
                bert_embed_dim: int = 768,
                feature_embed_dim: int = 512,
                freeze_layers:Union[Tuple[int, int], int] = None,
                text_head = False):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(bert_embed_dim, feature_embed_dim),
            nn.GELU(),
            nn.Linear(feature_embed_dim, feature_embed_dim)
        )
        self.text_head_bool = text_head
        if self.text_head_bool:
            self.text_head = nn.Sequential(
            nn.Linear(feature_embed_dim, feature_embed_dim),
            nn.GELU(),
            nn.Linear(feature_embed_dim, feature_embed_dim)
        )
        self.embed_dim = feature_embed_dim
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        # nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)
        
        if self.text_head_bool:
            for m in self.text_head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)#bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:",len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        if self.text_head_bool:
            encode_out = self.text_head(encode_out)
        return encode_out
    
    def forward(self,inpput_text):
        text_features = self.encode_text(inpput_text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features, None
    

class CLP_clinical(nn.Module):
    def __init__(self,
                bert_model_name: str,
                bert_embed_dim: int = 768,
                feature_embed_dim: int = 512,
                freeze_layers:Union[Tuple[int, int], int] = None
                ):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(bert_embed_dim, feature_embed_dim),
            nn.GELU(),
            nn.Linear(feature_embed_dim, feature_embed_dim)
        )
        self.embed_dim = feature_embed_dim
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        # nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)
        

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)#bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:",len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out
    
    def forward(self,inpput_text):
        text_features = self.encode_text(inpput_text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
