   
import os
import sys

from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import random
  
class GreedyBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        #print('self.drop_last',self.drop_last)
        #self.mosaic_array = list()

    '''
    def generate_mosaic_array(self):
        mosaic_array = []
        for i in range(self.batch_size):
            mosaic_array.append(random.choice([1,2,4]))
        return sum(self.mosaic_array)
    '''
    def get_random(self,sample):
        if random.random() < 0.5:
            num = random.choice(sample)
        else:
            num = 1
        return num
    def __iter__(self):
        batch = []
        sample = [2,3,4]
        num = self.get_random(sample)
        
        buckets = []
        for idx in self.sampler:
            buckets.append(idx)
            if len(buckets) == num :
                batch.append(buckets)
                num = self.get_random(sample)
                buckets = []
            if len(batch) == self.batch_size:
                yield batch
                #r,batch_size = self.get_random()
                #print('\n0-',batch_size)
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    #def get_mosaic_array(self) :
    #    return self.mosaic_array.pop(0)
    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.sampler)
        #if self.drop_last:
        #    return len(self.sampler) // self.batch_size  # type: ignore
        #else:
        #    return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore
