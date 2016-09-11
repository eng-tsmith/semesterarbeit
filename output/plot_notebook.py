from csv import reader
from matplotlib import pyplot
import matplotlib.patches as mpatches

with open('report.csv', 'r') as f:
    data = list(reader(f))

cost = [i[3] for i in data[1::]]
acc = [i[4] for i in data[1::]]

#  Cost / Acc over time
pyplot.plot(range(len(cost)), cost, 'r', range(len(acc)), acc, 'b')
#  legend
red_patch = mpatches.Patch(color='red', label='Cost')
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
pyplot.legend(handles=[blue_patch, red_patch])
#  Title and axis
pyplot.title('Cost & Accuracy over Time')
pyplot.xlabel('Iterations')
pyplot.ylabel('Cost CTC / Accuracy')
pyplot.show()
