from torch.utils.data import DataLoader
from data.data import UMLS_Dataset,DataInfo, PathKnowledge, PKDataset
from data.triplet_sampler import RandomIdentitySampler

def make_dataloader(args):

    num_workers = args.workers
    dataset = PathKnowledge(args.dataset_root)
    train_set = PKDataset(dataset.train)
    train_loader = DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instances),
                shuffle=False,
                num_workers=args.workers, 
                pin_memory=True,
            )

    if isinstance(dataset.test,dict):
        val_loader = {}
        len_query = {}
        for k,v in dataset.test.items():
            val_set = PKDataset(dataset.query[k] + dataset.gallery[k], istrain=False)
            val_loader[k] = DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
            )
            len_query[k] = len(dataset.query[k])
    elif isinstance(dataset.test,list):
        val_set = PKDataset(dataset.query + dataset.gallery, istrain=False)
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
        )
        len_query = len(dataset.query)

    return train_loader, val_loader, len(dataset.train), len_query
