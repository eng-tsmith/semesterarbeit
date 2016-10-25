from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.models import model_from_yaml
import numpy as np
import os
import IAM_pipeline.char_alphabet as char_alpha
import IAM_pipeline.preprocessor


def pad_sequence_into_array(image, maxlen):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    value = 0.
    image_ht = image.shape[0]

    Xout = np.ones(shape=[image_ht, maxlen], dtype=image[0].dtype) * np.asarray(value, dtype=image[0].dtype)

    trunc = image[:, :maxlen]

    Xout[:, :trunc.shape[1]] = trunc

    return Xout


def pad_label_with_blank(label, blank_id, max_length):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    label_len_1 = len(label[0])
    label_len_2 = len(label[0])

    label_pad = []
    # label_pad.append(blank_id)
    for _ in label[0]:
        label_pad.append(_)
        # label_pad.append(blank_id)

    while label_len_2 < max_length:
        label_pad.append(-1)
        label_len_2 += 1

    label_out = np.ones(shape=[max_length]) * np.asarray(blank_id)

    trunc = label_pad[:max_length]
    label_out[:len(trunc)] = trunc

    return label_out, label_len_1




# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Start Preprocessing
preprocessor   = IAM_pipeline.Preprocessor()

# Test image
path_to_img_file = '../media/nas/01_Datasets/IAM/words/c06/c06-005/c06-005-06-01.png'
path_to_xml = '../media/nas/01_Datasets/IAM/xml/c06-005.xml'
filename = 'c06-005-06-01'
input_tuple = [(path_to_img_file, path_to_xml, filename)]

#Preprocessing
preprocessor_output  = preprocessor.run(input_tuple, 0)

# Define batchsize
batch_size = len(preprocessor_output)
img_h = 64
img_w = 512
absolute_max_string_len = 100
chars = char_alpha.chars
output_size = len(chars)

pool_size_1 = 4
pool_size_2 = 2
downsampled_width = int(img_w / (pool_size_1 * pool_size_2) - 2)


# Define input shapes
if K.image_dim_ordering() == 'th':
    in1 = np.ones([batch_size, 1, img_h, img_w])
else:
    in1 = np.ones([batch_size, img_h, img_w, 1])
in2 = np.ones([batch_size, absolute_max_string_len])
in3 = np.zeros([batch_size, 1])
in4 = np.zeros([batch_size, 1])

# Define dummy output shape
out1 = np.zeros([batch_size])

# Pad/Cut all input to network size
for idx, input in enumerate(preprocessor_output):
    x_padded = pad_sequence_into_array(input[0], img_w)
    y_with_blank, y_len = pad_label_with_blank(np.asarray(input[1]), output_size,
                                               absolute_max_string_len)

    # Prepare input for model
    if K.image_dim_ordering() == 'th':
        # input_shape = (batchsize, 1, self.img_h, self.img_w)
        in1[idx, 0, :, :] = np.asarray(x_padded, dtype='float32')[:, :]
    else:
        # input_shape = (batchsize, self.img_h, self.img_w, 1)
        in1[idx, :, :, 0] = np.asarray(x_padded, dtype='float32')[:, :]
    in2[idx, :] = np.asarray(y_with_blank, dtype='float32')
    in3[idx, :] = np.array([downsampled_width], dtype='float32')
    in4[idx, :] = np.array([y_len], dtype='float32')

# Dictionary for Keras Model Input
inputs = {'the_input': in1,
          'the_labels': in2,
          'input_length': in3,
          'label_length': in4}
outputs = {'ctc': out1}

print(inputs)
print(outputs)
