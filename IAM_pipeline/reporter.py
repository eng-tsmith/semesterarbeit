from validation_task import ReporterTask
import csv


class IAM_Reporter(ReporterTask):
    def __init__(self):
        f = open("output/report.csv", "w")  #TODO Filename in config
        f.truncate()
        f.close()
        self.old_test = 0

        fields = ["Label", "Pred", "Loss", "CER", "WER", "Test Set:"]
        with open("output/report.csv", "a") as f:  #TODO Filename in config
            writer = csv.writer(f)
            writer.writerow(fields)

    def run(self, input_tuple, evaluator_output, test_set):
        """

        :param evaluator_output: [label, pred, loss, cer, wer]
        :param test_set:
        :return:
        """
        if test_set == 1:
            print("True label: ", evaluator_output[0])  # TODO include filename etc
            print("Prediction: ", evaluator_output[1])
            print("loss: ", evaluator_output[2])
            print("CER: ", evaluator_output[3])
            print("WER: ", evaluator_output[4])
            # ["Label", "Pred", "Loss", "CER", "WER", "Test Set:"]
            fields = [evaluator_output[0], evaluator_output[1], evaluator_output[2], evaluator_output[3], evaluator_output[4], test_set]
            with open("output/report.csv", "a") as f:  #TODO Filename in config
                writer = csv.writer(f)
                writer.writerow(fields)
        # put in 0 line between two testings
        if self.old_test == 0 and test_set == 1:
            self.old_test = test_set
        if self.old_test == 1 and test_set == 0:
            self.old_test = test_set
            fields = [0, 0, 0, 0, 0, 0]
            with open("output/report.csv", "a") as f:  # TODO Filename in config
                writer = csv.writer(f)
                writer.writerow(fields)

        return True

    def save(self, directory):
        print("Saving report to ", directory)
