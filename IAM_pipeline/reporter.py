from validation_task import ReporterTask
import csv


class IAM_Reporter(ReporterTask):
    def __init__(self):
        self.out_dir = "output/report.csv"  # TODO filename in config
        f = open(self.out_dir, "w")
        f.truncate()
        f.close()
        self.old_test = 0

        fields = ["File", "Label", "Pred", "Loss", "CE", "CER", "WER", "WER_LIB", "Test Set:"]
        with open(self.out_dir, "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def run(self, input_tuple, evaluator_output, test_set):
        """

        :param input_tuple:
        :param evaluator_output: [true, pred, loss, cer_abs, cer, wer, wer_lib]
        :param test_set:
        :return:
        """
        if test_set == 1:
            print("Filename: ", input_tuple[2])
            print("True label: ", evaluator_output[0])
            print("Prediction: ", evaluator_output[1])
            print("loss: ", evaluator_output[2])
            print("CE: ", evaluator_output[3])
            print("CER: ", evaluator_output[4])
            print("WER: ", evaluator_output[5])
            print("WER_lib: ", evaluator_output[6])
            # ["File", "Label", "Pred", "Loss", "CE", "CER", "WER", "WER_LIB", "Test Set:"]
            fields = [input_tuple[2], evaluator_output[0], evaluator_output[1], evaluator_output[2], evaluator_output[3], evaluator_output[4], evaluator_output[5],  evaluator_output[6], test_set]
            with open(self.out_dir, "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        # put in 0 line between two testings
        if self.old_test == 0 and test_set == 1:
            self.old_test = test_set
        if self.old_test == 1 and test_set == 0:
            self.old_test = test_set
            fields = [0, 0, 0, 0, 0, 0, 0]
            with open(self.out_dir, "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields)

        return True

    def save(self, directory):
        print("Saving report to ", directory)
