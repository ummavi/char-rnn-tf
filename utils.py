"""
Defines useful utility functions for dataset manipulations, etc.
"""
import collections
import numpy as np

class dataset:
    """
    Class to read a raw dataset (txt style block) and split them into batches of
    fixed sequence length.
    
    This code was taken and modified from:
    https://github.com/sherjilozair/char-rnn-tensorflow
    
    
    """
    def __init__(self,dataset_path,batch_size=64,seq_length=256):
        self.batch_size = batch_size
        self.seq_length = seq_length
        with open(dataset_path,"r") as f:
            self.data = f.read()
        self.create_batches()
        self.pointer = 0

        print("Initialized a dataset of",self.num_batches," batches ")

        
    def create_batches(self):
        counter = collections.Counter(self.data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocab.get, self.data)))

        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        
        
    def get_next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
