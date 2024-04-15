import torch
from torch import nn
import numpy as np

class AdaSPLoss(object):
    """
    SP loss using HARD example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, device, temp=0.04, loss_type = 'adasp'):
        self.device = device
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feats, targets):
        
        feat_q = nn.functional.normalize(feats, dim=1)
        
        bs_size = feat_q.size(0)
        N_id = len(torch.unique(targets))
        N_ins = bs_size // N_id

        scale = 1./self.temp

        sim_qq = torch.matmul(feat_q, feat_q.T)
        sf_sim_qq = sim_qq*scale

        right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).to(self.device)
        
        ## hard-hard mining for pos
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HH[mask_HH==0]=1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).to(self.device)
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH==0]=1.
        pos_sim_HH = 1./pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
        
        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
        
        ## hard-easy mining for pos
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HE[mask_HE==0]=1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE==0]=1.
        pos_sim_HE = 1./pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
        
    
        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
        wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 
    
        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).to(self.device)
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat,p = 1, dim = 1)

        loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adaptive = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
        
        if self.loss_type == 'sp-h':
            loss = loss_HH.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_HE.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adaptive
            
        return loss
        