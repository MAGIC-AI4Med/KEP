import torch
from PIL import Image
from KEP_model import load_KEP_model 


model_name = 'KEP_CTP_OpenPath'
model_bin_path = 'Path/to/model/root/' + model_name

device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
model,processor = load_KEP_model('ctp', model_name, model_bin_path, True, device)
model.eval()

example_image_path = './example.tif'
example_text = 'an H&E image of breast invasive carcinoma.'

img_input = processor['imgprocessor'](Image.open(example_image_path).convert('RGB')).unsqueeze(0).to(device)
token_input = processor['tokenizer']([example_text],max_length=256,padding='max_length',truncation=True, return_tensors='pt').to(device)

img_feature = model.encode_image(img_input).cpu().detach().numpy()
text_feature = model.encode_text(token_input).cpu().detach().numpy()
