
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

# def balanced_sample(data):
#     attr_num_dict = dict()
#     for idx in data:
#         if idx[1] not in attr_num_dict:
#             attr_num_dict[idx[1]] = [idx]
#         else:
#             attr_num_dict[idx[1]].append(idx)
    
#     for k,v in attr_num_dict.items():

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (text, did, tid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_dids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, did, _, attr) in enumerate(self.data_source):
            self.index_dic[did].append([index, attr])
        self.dids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for did in self.dids:
            idxs = self.index_dic[did]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for did in self.dids:
            idxs = copy.deepcopy(self.index_dic[did])
            if len(idxs) < self.num_instances:
                sample_index = np.random.choice(list(range(len(idxs))), size=self.num_instances-len(idxs), replace=True)
                for i in list(sample_index):
                    idxs.append(idxs[i])
                random.shuffle(idxs)
                batch_idxs_dict[did].append([idx[0] for idx in idxs])

            ## each id contains at least main
            else:
                random.shuffle(idxs)
                for idx in idxs:
                    if idx[1] == 'main':
                        main_idx = idx[0]
                        break
                batch_idxs = []
                for idx in idxs:
                    if idx[1] == 'main':
                        continue
                    batch_idxs.append(idx[0])
                    if len(batch_idxs) == self.num_instances-1:
                        batch_idxs.append(main_idx)
                        batch_idxs_dict[did].append(batch_idxs)
                        batch_idxs = []

                ## don't discard the rest
                if len(batch_idxs) > 0:
                    batch_idxs.append(main_idx)
                    sample_index = np.random.choice(batch_idxs, size=self.num_instances-len(batch_idxs), replace=True)
                    batch_idxs += list(sample_index)
                    random.shuffle(batch_idxs)
                    batch_idxs_dict[did].append(batch_idxs)


        avai_dids = copy.deepcopy(self.dids)
        final_idxs = []

        while len(avai_dids) >= self.num_dids_per_batch:
            selected_dids = random.sample(avai_dids, self.num_dids_per_batch)
            for did in selected_dids:
                batch_idxs = batch_idxs_dict[did].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[did]) == 0:
                    avai_dids.remove(did)

        return iter(final_idxs)

    def __len__(self):
        return self.length