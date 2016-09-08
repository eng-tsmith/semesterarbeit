# __________________________
# _________ OUTPUT _________
# __________________________

outdir = "/results/"
outfile = 'IAM_output_NN.txt'

# ___________________________
# ___________ IAM ___________
# ___________________________

# Settings
apply_histeq = False
apply_liver_crf = True
use_net2 = True #for CFCN
net1_n_classes = 2

# Path TODO: think about good ways
# IAM_BASE_PATH = "../../Datasets/IAM/"
IAM_BASE_PATH = "../media/nas/01_Datasets/IAM/"
IAM_img_path = IAM_BASE_PATH + "lines/"
IAM_label_path =IAM_BASE_PATH + "xml/"

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

IAM_dataset_train = []
IAM_dataset_validate = []

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
        label = IAM_label_path + part1 + "-" + part2 + ".xml"
        set.append((image, label, name))
    IAM_dataset_validate.append(set)

IAM_bs = "IAM_bs" # TODO
IAM_models = [IAM_bs + "_model"]  # TODO


IAM = [IAM_dataset_train, IAM_dataset_validate, IAM_models]

# ____________________________
# ______ SELECT DATASET ______
# ____________________________

# dataset_train, dataset_val, models = IAM
dataset_train, models = IAM







