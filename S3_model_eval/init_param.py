import torch

def param_init(model_name):

    params = dict()
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model address
    params['bert_path'] = '../pretrained_model/pubmedbert'
    params['biobert_path'] = '../pretrained_model/Bio_ClinicalBERT'
    params['pubbert_path'] = '../pretrained_model/pubmedbert'
    
    ## models pretrained on OpenPath
    KEP_32_OpenPath = '../model_zoo/KEP_32_OpenPath/checkpoints/epoch_30.pt'
    KEP_16_OpenPath = '../model_zoo/KEP_16_OpenPath/checkpoints/epoch_30.pt'
    KEP_CTP_OpenPath = '../model_zoo/KEP_CTP_OpenPath/checkpoints/epoch_30.pt'  
    
    ## models pretrained on Qilt1m
    KEP_32_Quilt1m = '../model_zoo/KEP_32_Quilt1m/checkpoints/epoch_15.pt'
    KEP_16_Quilt1m = '../model_zoo/KEP_16_Quilt1m/checkpoints/epoch_15.pt'
    KEP_CTP_Quilt1m = '../model_zoo/KEP_CTP_Quilt1m/checkpoints/epoch_15.pt'
    
    params['model_name'] = model_name
    params['arch_name'] = 'ViT-B/32' if '32' in model_name else 'ViT-B/16'
    params['model_path'] = eval(params['model_name'])
    params['model_type'] = 'vit_bert'  # 'biomed_bert' for 'ViT-B/16'
    params['max_token'] = 256
    params['visual_head']  =True
   
    return params
