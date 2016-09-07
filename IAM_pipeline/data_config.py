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

# Line Name
IAM_name_1 = "a01-000u-00"
IAM_name_2 = "a01-000u-01"

IAM_validation_set = [
    (IAM_img_path+"a01/a01-000u/"+IAM_name_1+".png", IAM_label_path+"a01-000u.xml", IAM_name_1),
    (IAM_img_path+"a01/a01-000u/"+IAM_name_2+".png", IAM_label_path+"a01-000u.xml", IAM_name_2)]


IAM_dataset = [IAM_validation_set]  # TODO
IAM_bs = "IAM_bs" # TODO
IAM_models = [IAM_bs + "_model"]  # TODO


IAM = [IAM_dataset, IAM_models]

# ____________________________
# ______ SELECT DATASET ______
# ____________________________

dataset, models = IAM







