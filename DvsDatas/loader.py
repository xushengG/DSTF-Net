import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers, pin_memory=pin_memory,
                                             collate_fn=collate_events, 
                                             drop_last=True)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    fus = []
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[2])
        fus.append(d[0])
        ev = np.concatenate([i*np.ones((len(d[1]),1), dtype=np.float32), d[1]],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    fus = default_collate(fus)
    return fus, events, labels