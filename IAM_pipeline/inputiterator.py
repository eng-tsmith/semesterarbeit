from validation_task import InputIteratorTask
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print "====== IAM  Pipeline ======"

        inputs = [np.random.random((1,3)), np.random.random((1,3)), np.random.random((1,3))]
        for i in inputs:
            yield i

    def __len__(self):
        fold_lens = 3  # TODO: old --> map(lambda fold: len(fold), IAM_config.dataset)
        return fold_lens
