from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")

        for fold in data.dataset:
            for input in fold:
                print("test:", input)
                yield input

    def __len__(self):
        fold_lens = map(lambda fold: len(fold), data.dataset)
        return functools.reduce(lambda a, b: a + b, fold_lens)
