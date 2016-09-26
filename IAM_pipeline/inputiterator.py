from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")

        for epoch in range(100):  # TODO fix in config
            print("Epoche: ", epoch)
            # for fold in data.dataset_words: # TODO WORD!!!
            #     for input in fold:
            #         print("Train with: ", input)
            #         yield input, 0

            for fold in data.dataset_val_words:  #TODO
                for input in fold:
                    print("Test:", input)
                    yield input, 1

    def __len__(self):
        fold_lens1 = map(lambda fold: len(fold), data.dataset_train)
        fold_lens2 = map(lambda fold: len(fold), data.dataset_val)

        return (functools.reduce(lambda a, b: a + b, fold_lens1) + functools.reduce(lambda a, b: a + b,
                                                                                    fold_lens2)) * 100  # TODO fix in config
