from validation_task import InputIteratorTask
import data_config
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print "====== IAM  Pipeline ======"

        for img_label in data_config.dataset:
            yield img_label

    def __len__(self):
        fold_lens = 3  # TODO: old --> map(lambda fold: len(fold), IAM_config.dataset)
        return fold_lens
