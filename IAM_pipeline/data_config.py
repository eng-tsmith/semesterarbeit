import xml.etree.ElementTree as ET
# __________________________
# _________ OUTPUT _________
# __________________________

outdir = "/results/"
outfile = 'IAM_output_NN.txt'


# ___________________________
# ___________ IAM ___________
# ___________________________

IAM_BASE_PATH = "../media/nas/01_Datasets/IAM/"
IAM_img_path = IAM_BASE_PATH + "lines/"
IAM_label_path =IAM_BASE_PATH + "xml/"
IAM_word_path = IAM_BASE_PATH + "words/"

# Divide datset into sets given by IAM
trainset_path = '../media/nas/01_Datasets/IAM/trainset.txt'
valset1_path = '../media/nas/01_Datasets/IAM/validationset1.txt'
valset2_path = '../media/nas/01_Datasets/IAM/validationset2.txt'
testset_path = '../media/nas/01_Datasets/IAM/testset.txt'
timset_path = '../media/nas/01_Datasets/IAM/tim_set.txt'



# This determins which sets are used for training
# files_training = [timset_path]
files_training = [trainset_path]
files_validate = [valset1_path]
files_words = [valset1_path]

IAM_dataset_train = []
IAM_dataset_validate = []
IAM_dataset_words = []

for path in files_training:
    with open(path, 'r') as txtfile:
        content = txtfile.readlines()
    set =[]
    for row in content:
        part1 = row.split('-')[0]
        part2 = row.split('-')[1]
        name = row.split('\n')[0]
        image = IAM_img_path + part1 + "/" + part1 + "-" + part2 + "/" + name + ".png"
        label = IAM_label_path + part1 + "-" + part2 + ".xml"
        set.append((image, label, name))
    IAM_dataset_train.append(set)

for path in files_validate:
    with open(path, 'r') as txtfile:
        content = txtfile.readlines()
    set =[]
    for row in content:
        part1 = row.split('-')[0]
        part2 = row.split('-')[1]
        name = row.split('\n')[0]
        image = IAM_img_path + part1 + "/" + part1 + "-" + part2 + "/" + name + ".png"
        label = IAM_label_path + part1 + "-" + part2 + ".xml"  # TODO  exclude label
        set.append((image, label, name))
    IAM_dataset_validate.append(set)

for path in files_words:
    with open(path, 'r') as txtfile:
        content = txtfile.readlines()

    for row in content:
        part1 = row.split('-')[0]
        part2 = row.split('-')[1]
        label = IAM_label_path + part1 + "-" + part2 + ".xml"

        filename = row.split('\n')[0]
        filepath = IAM_label_path + filename + ".xml"

        tree = ET.parse(label)
        root = tree.getroot()
        set = []
        for child in root.iter("word"):
            # if child.get('id') == filename:
            image = IAM_word_path + part1 + "/" + part1 + "-" + part2 + "/" + child.get('id') + ".png"
            label = IAM_label_path + part1 + "-" + part2 + ".xml"
            set.append((image, label, child.get('id')))
        IAM_dataset_words.append(set)

IAM = [IAM_dataset_train, IAM_dataset_validate, IAM_dataset_words]

# ____________________________
# ______ SELECT DATASET ______
# ____________________________

dataset_train, dataset_val, dataset_words = IAM







