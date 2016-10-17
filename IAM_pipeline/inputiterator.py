from validation_task import InputIteratorTask
import IAM_pipeline.data_config as data
import functools


class IAM_InputIterator(InputIteratorTask):
    def run(self):
        print("Welcome to Handwriting Recognizer")
        print("====== IAM  Pipeline ======")
        # Init Epochs and Batchsize
        n_epochs_word = data.n_epochs_word
        n_epochs_line = data.n_epochs_line
        n_batch_size = data.n_batch_size

        print("====== Word Training ======")
        for epoch in range(1, n_epochs_word + 1):
            print("Epoche: ", epoch)
            for fold in data.dataset_words:
                inputs = []
                for input in fold:
                    # print("Train with: ", input)
                    inputs.append(input)
                    if len(inputs)== n_batch_size:
                        input_batch = inputs
                        inputs = []
                        yield input_batch, 0, 0, epoch

            for fold in data.dataset_val_words:
                inputs = []
                for input in fold:
                    # print("Test:", input)
                    inputs.append(input)
                    if len(inputs) == n_batch_size:
                        input_batch = inputs
                        inputs = []
                        yield input_batch, 1, 0, epoch

        print("====== Line Training ======")
        for epoch in range(1, n_epochs_line + 1):
            print("Epoche: ", epoch)
            for fold in data.dataset_train:
                inputs = []
                for input in fold:
                    # print("Train with: ", input)
                    inputs.append(input)
                    if len(inputs) == n_batch_size:
                        input_batch = inputs
                        inputs = []
                        yield input_batch, 0, 1, epoch

            for fold in data.dataset_val:
                inputs = []
                for input in fold:
                    # print("Test:", input)
                    inputs.append(input)
                    if len(inputs) == n_batch_size:
                        input_batch = inputs
                        inputs = []
                        yield input_batch, 1, 1, epoch

    def sizes(self):  # TODO
        fold_lens1 = data.dataset_words_size
        fold_lens2 = data.dataset_val_words_size
        fold_lens3 = data.dataset_train_size
        fold_lens4 = data.dataset_val_size

        n_epochs_word = data.n_epochs_word
        n_epochs_line = data.n_epochs_line

        return fold_lens1, fold_lens2, fold_lens3, fold_lens4, n_epochs_word, n_epochs_line
