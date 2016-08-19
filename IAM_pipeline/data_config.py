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
# Path
IAM_BASE_PATH = "../../Datasets/IAM/"
IAM_img_path = IAM_BASE_PATH + "lines/"
IAM_label_path = IAM_BASE_PATH + "xml/"

IAM_validation_set = [
    IAM_img_path+"a01/a01-000u/a01-000u-00.png", IAM_label_path+"a01-000u.xml"]

IAM_dataset = [IAM_validation_set]  # TODO
IAM_bs = "IAM_bs" # TODO
IAM_models = [IAM_bs + "_model"]  # TODO


IAM = [IAM_dataset, IAM_models]

# ____________________________
# ______ SELECT DATASET ______
# ____________________________

dataset, models = IAM







