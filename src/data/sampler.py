from typing import Iterator, List, Dict
from torch.utils.data import Dataset
import numpy as np
import random
from torch.utils.data.sampler import Sampler
import copy


class ByWriterIDSampler(Sampler):
    def __init__(self, data_source: Dataset, shuffle, batch_size, drop_last=False) -> None:  # type: ignore[type-arg]
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.all_cluster_indices = self.data_source._cluster_wid_indices()  # type: ignore[attr-defined]
        self.corpus_size = sum([len(cluster) for cluster in self.all_cluster_indices.values()])
        self.wids = list(self.all_cluster_indices.keys())

    def sampling_batch_ids(self, cur_wids, cur_cluster_indices: Dict[str, List[int]]):
        cur_num_wid = len(cur_wids)
        cur_batch_size = min(cur_num_wid, self.batch_size)

        cur_dist = np.array([len(cur_cluster_indices[wid]) for wid in cur_wids.tolist()])
        sampling_weight = cur_dist / cur_dist.sum()
        choose_wids = np.random.choice(cur_wids, size=cur_batch_size, replace=False, p=sampling_weight).tolist()

        batch_ids = []
        for wid in choose_wids:
            item_idx = random.choice(cur_cluster_indices[wid])
            batch_ids.append(item_idx)
            cur_cluster_indices[wid].remove(item_idx)
            if len(cur_cluster_indices[wid]) == 0:
                cur_cluster_indices.pop(wid)

        return batch_ids

    def build_batch_ids(self):
        cur_corpus_size = self.corpus_size
        cur_cluster_indices = copy.deepcopy(self.all_cluster_indices)

        total_batch_ids = []
        while cur_corpus_size > 0:
            # Check remain wids list after sampling
            cur_wids = np.array(list(cur_cluster_indices.keys()))
            if self.drop_last and len(cur_wids) < 2:
                break

            if len(cur_wids) < 2:
                # This is the edge case
                # Handle remain sample which belong to only one wids, so in order to form a batch, we
                # randomly choose a sample from other wids list, which have already use in other batches
                # in other word, these samples are appear more than one time in the training progress
                wid = cur_wids.tolist()[0]
                self.wids.remove(wid)
                for item_idx in cur_cluster_indices[wid]:
                    random_wid = random.choice(self.wids)
                    pair_idx = random.choice(self.all_cluster_indices[random_wid])
                    batch_ids = [item_idx, pair_idx]
                    total_batch_ids.append(batch_ids)
                break
            else:
                batch_ids = self.sampling_batch_ids(cur_wids, cur_cluster_indices)

            cur_corpus_size = sum([len(cluster) for cluster in cur_cluster_indices.values()])
            total_batch_ids.append(batch_ids)

        if self.shuffle:
            random.shuffle(total_batch_ids)

        return total_batch_ids

    def __iter__(self) -> Iterator:
        for batch_id in self.build_batch_ids():
            yield batch_id
