import os
import warnings
warnings.filterwarnings("ignore")
from evaluate import Evaluater
from init_param import param_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    # dataset address
    dataset_root = '../dataset/test_dataset/'
    bach_test_img_root = os.path.join(dataset_root, 'Bach', 'images_224')
    Kather_train_img_root = os.path.join(dataset_root, 'Kather', 'NCT-CRC-HE-100K-NONORM')
    Kather_test_img_root = os.path.join(dataset_root, 'Kather', 'CRC-VAL-HE-7K')
    lc25000_test_img_root = os.path.join(dataset_root, 'Lc25000', 'images_224')
    renalcell_test_img_root = os.path.join(dataset_root, 'Renalcell', 'images_224')    
    sicap_test_img_root = os.path.join(dataset_root, 'Sicap', 'images_224')
    skincancer_test_img_root = os.path.join(dataset_root, 'Skincancer', 'images_224')
    wss_test_img_root = os.path.join(dataset_root, 'Wsss4luad', 'wsss4luad')


    abook_test_img_root = os.path.join(dataset_root, 'Arch_book', 'arch_book_retrieval')
    apubmed_test_img_root = os.path.join(dataset_root, 'Arch_pubmed', 'arch_pubmed_retrieval')


    image_key = 'image_name'
    caption_key = 'caption'
    label_key = 'label'

    # model to evaluate
    model_name = 'KEP-32_OpenPath'
    model_params = param_init(model_name)

    evaluater = Evaluater(model_params)
    evaluater.load_model(model_params)

    # zero shot evaluation
    print('*************************************************************************')
    print('Zeroshot classification on Bach test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Bach','test', 
                            bach_test_img_root, image_key, caption_key, label_key)

    print('*************************************************************************')
    print('Zeroshot classification on Kather train datasets...')
    evaluater.zeroshot_eval(dataset_root,'Kather','train', 
                            Kather_train_img_root, image_key, caption_key, label_key, sep = ',')

    print('*************************************************************************')
    print('Zeroshot classification on Kather test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Kather','test', 
                            Kather_test_img_root, image_key, caption_key, label_key, sep =',')
    
    print('*************************************************************************')
    print('Zeroshot classification on Lc25000 test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Lc25000','test', 
                            lc25000_test_img_root, image_key, caption_key, label_key)
    
    print('*************************************************************************')
    print('Zeroshot classification on Renalcell test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Renalcell','test', 
                            renalcell_test_img_root, image_key, caption_key, label_key)
    
    print('*************************************************************************')
    print('Zeroshot classification on Sicap test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Sicap','test', 
                            sicap_test_img_root, image_key, caption_key, label_key)
    
    print('*************************************************************************')
    print('Zeroshot classification on Skincancer test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Skincancer','test', 
                            skincancer_test_img_root, image_key, caption_key, label_key)
    
    print('*************************************************************************')
    print('Zeroshot classification on Wsss4luad test datasets...')
    evaluater.zeroshot_eval(dataset_root,'Wsss4luad','binary_test', 
                            wss_test_img_root, image_key, caption_key, label_key, sep = ',')
    
    
     # text2img retrieval
    print('*************************************************************************')
    print('text2img retrieval on arch_pubmed datasets...')
    evaluater.text2img_retrieval_eval(dataset_root,'Arch_pubmed','test', 
                                      apubmed_test_img_root, image_key, caption_key)

    print('*************************************************************************')
    print('text2img retrieval on arch_book datasets...')
    evaluater.text2img_retrieval_eval(dataset_root,'Arch_book','test', 
                                      abook_test_img_root, image_key, caption_key)


    # img2text retrieval
    print('*************************************************************************')
    print('img2text retrieval on arch_pubmed datasets...')
    evaluater.img2text_retrieval_eval(dataset_root,'Arch_pubmed','test', 
                                      apubmed_test_img_root, image_key, caption_key)

    print('*************************************************************************')
    print('img2text retrieval on arch_book datasets...')
    evaluater.img2text_retrieval_eval(dataset_root,'Arch_book','test', 
                                      abook_test_img_root, image_key, caption_key)

    