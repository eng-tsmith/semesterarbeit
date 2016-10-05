from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools
import numpy as np


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")
        n_epochs_word = data.n_epochs_word
        n_epochs_line = data.n_epochs_line

        print("====== Word Training ======")
        for epoch in range(n_epochs_word):
            print("Epoche: ", epoch)
            # for fold in data.dataset_words:
            #     for input in fold:
            #         print("Train with: ", input)
            #         yield input, 0

            for fold in data.dataset_val_words:
                for input in fold:
                    print("Test:", input)
                    yield input, 1

        print("====== Line Training ======")
        for epoch in range(n_epochs_line):
            print("Epoche: ", epoch)
            # for fold in data.dataset_train:
            #     for input in fold:
            #         print("Train with: ", input)
            #         yield input, 0

            for fold in data.dataset_val:
                for input in fold:
                    print("Test:", input)
                    yield input, 1

    def __len__(self):
        fold_lens1 = map(lambda fold: len(fold), data.dataset_train)
        fold_lens2 = map(lambda fold: len(fold), data.dataset_val)

        return (functools.reduce(lambda a, b: a + b, fold_lens1) + functools.reduce(lambda a, b: a + b,
                                                                                    fold_lens2)) * 100  # TODO fix in config
