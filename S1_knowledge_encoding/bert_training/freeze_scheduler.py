
import logging
import copy
import torch

class FreezeScheduler():
    def __init__(self,):
        self.freeze_flag = False
    
    def update(self, freeze_bert_epochs, model, epoch):

        # freese text encoder
        if epoch < freeze_bert_epochs and not self.freeze_flag:
            logging.info('Freeze bert')
            for name, param in model.bert_model.named_parameters():
                param.requires_grad = False
            self.freeze_flag = True

        elif epoch >= freeze_bert_epochs and self.freeze_flag:
            logging.info('Open bert')
            for name, param in model.bert_model.named_parameters():
                param.requires_grad = True
            self.freeze_flag = False


class FreezeChecker():
    def __init__(self, model):
        self.before = dict()
        for name, kid in model.named_children():
            self.before[name] = {}
            for subname in kid.state_dict():
                if 'weight' in subname:
                    self.before[name][subname] =  copy.deepcopy(kid.state_dict()[subname])
                    break

    def freeze_checker(self, model):
        weight_diff = dict()
        for name, kid in model.named_children():
            if name in self.before:
                weight_diff[name] = 0.
                for subname in kid.state_dict():
                    if subname in self.before[name]:
                        weight_diff[name] =  torch.norm(self.before[name][subname] - kid.state_dict()[subname],p=2)
                        if weight_diff[name] > 0:
                            logging.info(f'{name} module is open.')
                        else:
                            logging.info(f'{name} module is freezed.')
                        break

