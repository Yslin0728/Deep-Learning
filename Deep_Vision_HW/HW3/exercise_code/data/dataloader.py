"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items(): 
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict
        
        def batch_to_numpy(batch):
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch
        
        
        def build_batch_iterator(dataset, batch_size, shuffle):
            if shuffle:
                index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
            else:
                index_iterator = iter(range(len(dataset)))  # define indices as iterator
            
            batch = []
            count = 1
            ind = (len(dataset)/batch_size) #dataset可以裝滿幾個batch
            
            for index in index_iterator:  # iterate over indices using the iterator
                batch.append(dataset[index])
                if self.drop_last:
                    if len(batch) == batch_size:
                        yield batch  # use yield keyword to define a iterable generator
                        batch = []
                else:
                    if len(batch) == batch_size:
                        yield batch  # use yield keyword to define a iterable generator
                        batch = []
                        
                    elif (count/batch_size == ind and len(batch) < batch_size): #如果batch已經是最後一個裝滿的情況且小於batch size
                        yield batch  # use yield keyword to define a iterable generator
                        batch = []
                        
                    count = count +1    
                
        batch_iterator = build_batch_iterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        batches = []
        for batch in batch_iterator:
            batches.append(batch)
            
        
        combined_batches = [combine_batch_dicts(batch) for batch in batches]
        numpy_batches = [batch_to_numpy(batch) for batch in combined_batches]
        
        return iter(numpy_batches)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################
        
        length = int(len(self.dataset) / self.batch_size)
        
        if (self.drop_last == False):
            if(len(self.dataset) % self.batch_size != 0):
                length = length + 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
