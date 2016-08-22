from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print ("====== IAM  Pipeline ======")

        for img_label in data.dataset:
            yield img_label

    def __len__(self):
        fold_lens = 3  # TODO: old --> map(lambda fold: len(fold), IAM_config.dataset)
        return fold_lens
