from validation_task import ReporterTask
import csv


class IAM_Reporter(ReporterTask):
    def __init__(self):
        f = open("output/report.csv", "w")  #TODO Filename in config
        f.truncate()
        f.close()

        fields = ["Label", "Loss", "Metric", "Test Set:"]
        with open("output/report.csv", "a") as f:  #TODO Filename in config
            writer = csv.writer(f)
            writer.writerow(fields)

    def run(self, evaluator_output, test_set):
        """

        :param input_tuple:
        :param postprocessor_output: Predictions, Cost, Shown String, Seen String
        :param evaluator_output: match, accuracy
        :return:
        """
        print("Loss: ", evaluator_output[1], "\n",
              "Metric: ", evaluator_output[2], "\n",
              "Test Set: ", test_set, "\n")

        fields = [evaluator_output[0], evaluator_output[1], evaluator_output[2], test_set]

        with open("output/report.csv", "a") as f:  #TODO Filename in config
            writer = csv.writer(f)
            writer.writerow(fields)

        return True

    def save(self, directory):
        print ("Saving report to ", directory)
