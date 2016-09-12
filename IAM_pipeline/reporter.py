from validation_task import ReporterTask
import csv


class IAM_Reporter(ReporterTask):
    def __init__(self):
        f = open("output/report.csv", "w")  #TODO Filename in config
        f.truncate()
        f.close()

        fields = ["Match", "Shown", "Seen", "Cost", "Total Accuracy:", "Test Set:"]
        with open("output/report.csv", "a") as f:  #TODO Filename in config
            writer = csv.writer(f)
            writer.writerow(fields)

    def run(self, input_tuple, postprocessor_output, evaluator_output, test_set):
        """

        :param input_tuple:
        :param postprocessor_output: Predictions, Cost, Shown String, Seen String
        :param evaluator_output: match, accuracy
        :return:
        """
        print("Match?", evaluator_output[0], "Shown: ", ''.join(postprocessor_output[2]), "Seen: ", ''.join(postprocessor_output[3]),
              "Cost: ", postprocessor_output[1], "Total Accuracy: ", evaluator_output[1], "Test Set: ", test_set)

        fields = [evaluator_output[0], postprocessor_output[2], postprocessor_output[3], postprocessor_output[1],
                  evaluator_output[1], test_set]

        with open("output/report.csv", "a") as f:  #TODO Filename in config
            writer = csv.writer(f)
            writer.writerow(fields)

        return True

    def save(self, directory):
        print ("Saving report to ", directory)
