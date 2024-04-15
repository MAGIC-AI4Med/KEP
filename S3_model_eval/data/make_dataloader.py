from torch.utils.data import DataLoader
from data.data import DataInfo, PathKnowledge, PKDataset
from data.triplet_sampler import RandomIdentitySampler

def make_dataloader(dataset_root, workers, batch_size):

    num_workers = workers
    dataset = PathKnowledge(dataset_root)
   
    if isinstance(dataset.test,dict):
        val_loader = {}
        len_query = {}
        for k,v in dataset.test.items():
            val_set = PKDataset(dataset.query[k] + dataset.gallery[k])
            val_loader[k] = DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            len_query[k] = len(dataset.query[k])
    elif isinstance(dataset.test,list):
        val_set = PKDataset(dataset.query + dataset.gallery)
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        len_query = len(dataset.query)

    return val_loader, len_query
