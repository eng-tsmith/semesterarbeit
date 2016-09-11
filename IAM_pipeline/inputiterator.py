from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")

        for epoch in range(100):
            print("Epoche: ", epoch)
            for fold in data.dataset_train:
                for input in fold:
                    print("Train with: ", input)
                    yield input, 0

            for fold in data.dataset_val:  #TODO
                for input in fold:
                    print("Test:", input)
                    yield input + [1]

    def __len__(self):
        fold_lens = map(lambda fold: len(fold), data.dataset_train)
        return functools.reduce(lambda a, b: a + b, fold_lens)
