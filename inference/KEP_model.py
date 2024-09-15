from typing import Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model_transformer import LayerNormFp32, LayerNorm, QuickGELU, VisionTransformer
from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel,BertConfig, AutoTokenizer  ## transformers >= 4.34.0
import timm_ctp
from timm_ctp.models.layers.helpers import to_2tuple
from torchvision.transforms import Normalize, InterpolationMode, ToTensor, Resize, CenterCrop, Compose


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    

def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual


class PATH_BERT(nn.Module):
    def __init__(self,
                bert_model_name: str,
                bert_embed_dim: int = 768,
                feature_dim: int = 512,
                freeze_layers:Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(bert_embed_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.embed_dim = feature_dim
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

class KEP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            image_encoder: str,
            bert_pretrain: str,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            visual_embedding_head: bool = False,
            text_embedding_head: bool = False,
            logit_scale = 0.07
    ):
        super().__init__()
        self.output_dict = output_dict
        self.bert_pretrain = bert_pretrain
        self.image_encoder = image_encoder

        self.visual_embedding_head = visual_embedding_head
        self.text_embedding_head = text_embedding_head
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        if image_encoder == 'res_ssl':
            self.pool_layer = nn.AvgPool2d((7, 7))
        if visual_embedding_head:
            if image_encoder in ['ctp','res_pmc']:
                input_dim = 768
            elif image_encoder == 'res_ssl':
                input_dim = 2048
            else:
                input_dim = 512
            self.visual_head = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for m in self.visual_head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

        self.text = PATH_BERT(bert_model_name = self.bert_pretrain, feature_dim = embed_dim)
        if text_embedding_head:
            self.text_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for m in self.text_head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))

    def encode_image(self, image, normalize: bool = False):
        if self.image_encoder == 'res_pmc':
            features = self.visual(image)['image_features']
        else:
            features = self.visual(image)
        if self.image_encoder == 'res_ssl':
            features = self.pool_layer(features).squeeze()
        if self.visual_embedding_head:
            features = self.visual_head(features)
        return F.normalize(features, dim=-1) if normalize else features


    def encode_text(self, text, normalize: bool = False):
        x = self.text(text)
        if self.text_embedding_head:
            x = self.text_head(x)
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
            
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
            }
        return image_features, text_features


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath(img_size = 224, **kwargs):
    model = timm_ctp.create_model('swin_tiny_patch4_window7_224', 
                                  embed_layer=ConvStem, 
                                  pretrained=False,
                                  img_size=img_size,
                                  **kwargs)
    return model

def load_KEP_model(arch_name, model_name, model_path, visual_head, device):
    arch_name = arch_name.replace('/','-')
    if arch_name.lower() =='vit-b-32':
        vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32}
    elif arch_name.lower() =='vit-b-16':
        vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 16}
    else:
        vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 16}
          
    cast_dtype = get_cast_dtype('amp')
    image_encoder = 'vit'
    if 'CTP' in model_name:
        image_encoder = 'ctp'
    model = KEP(embed_dim=512,
                vision_cfg= vision_cfg,
                image_encoder=image_encoder,
                bert_pretrain= model_path, 
                cast_dtype=cast_dtype, 
                visual_embedding_head=visual_head,
                )

    # model_root = '../pretrained_model/CTransPath/ctranspath.pth'
    ctp_model = ctranspath()
    ctp_model.head = nn.Identity()
    # ctp_model.head = nn.Identity()
    # state_dict = torch.load(model_root, map_location="cpu")
    # missing_keys, unexpected_keys = ctp_model.load_state_dict(state_dict['model'], strict=False)
    # print('missing keys: ', missing_keys)
    # print('unexpected keys: ', unexpected_keys)

    model.visual = ctp_model
    model.visual.image_size = 224
    model.visual.image_mean = (0.485, 0.456, 0.406)
    model.visual.image_std = (0.229, 0.224, 0.225)
    # print('Load pretrained vision encoder success from CTransPath.')
    
    checkpoint = torch.load(model_path + '/pytorch_model.bin', map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict,strict=True)
    model.to(device)
    model.eval()

    processor = {}
    processor['tokenizer'] = AutoTokenizer.from_pretrained(model_path,do_lower_case=True, local_files_only=True)
    img_mean = getattr(model.visual, 'image_mean', None)
    img_std = getattr(model.visual, 'image_std', None)
    
    processor['imgprocessor'] = Compose([Resize(model.visual.image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(model.visual.image_size),
            ToTensor(),
            Normalize(img_mean, img_std),
        ])

    return model,processor