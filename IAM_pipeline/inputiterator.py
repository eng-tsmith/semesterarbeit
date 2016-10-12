from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")
        n_epochs_word = data.n_epochs_word
        n_epochs_line = data.n_epochs_line

        print("====== Word Training ======")
        for epoch in range(n_epochs_word):
            print("Epoche: ", epoch)
            for fold in data.dataset_words:
                for input in fold:
                    print("Train with: ", input)
                    yield input, 0, 0, epoch

            for fold in data.dataset_val_words:
                for input in fold:
                    print("Test:", input)
                    yield input, 1, 0, epoch

        print("====== Line Training ======")
        for epoch in range(n_epochs_line):
            print("Epoche: ", epoch)
            for fold in data.dataset_train:
                for input in fold:
                    print("Train with: ", input)
                    yield input, 0, 1, epoch

            for fold in data.dataset_val:
                for input in fold:
                    print("Test:", input)
                    yield input, 1, 1, epoch

    def __len__(self):
        fold_lens1 = len(data.dataset_words)  #map(lambda fold: len(fold), data.dataset_words)
        fold_lens2 = len(data.dataset_val_words)  #map(lambda fold: len(fold), data.dataset_val_words)
        fold_lens3 = len(data.dataset_train)  #map(lambda fold: len(fold), data.dataset_train)
        fold_lens4 = len(data.dataset_val)  #map(lambda fold: len(fold), data.dataset_val)

        return fold_lens1, fold_lens2, fold_lens3, fold_lens4
