from keras.caffe import convert
from scipy.misc import imread, imresize
import numpy as np

cnn_model_def = "../bvlc_googlenet.prototxt"
# cnn_model_def = "../train_val.prototxt"
cnn_model_params = "../bvlc_googlenet.caffemodel"

C = 3
H = 224
W = 224

def format_img_for_input(image, H, W):
    """
    Helper function to convert image read from imread to caffe input

    Input:
    image - numpy array describing the image
    H - height in px
    W - width in px
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    # RGB -> BGR
    image = image[:, :, (2, 1, 0)]
    # mean subtraction (get mean from model file?..hardcoded for now)
    image = image - np.array([103.939, 116.779, 123.68])
    # resize
    image = imresize(image, (H, W))
    # get channel in correct dimension
    image = np.transpose(image, (2, 0, 1))
    return image

print "Loading model"
model = convert.caffe_to_keras(
    prototext = cnn_model_def,
    caffemodel = cnn_model_params,
    phase = 'train',
)
graph = model

print "Compiling"
graph.compile('rmsprop', {graph.outputs.keys()[0]: 'mse'})

keras_batch = np.zeros((1, C, H, W))

# Load image and format for input
print "Loading example image..."
im = imread('../cat.png')
formatted_im = format_img_for_input(im, H, W)
keras_batch[0, :, :, :] = formatted_im

print "Extracting features from keras Graph..."
keras_features = graph.predict({'conv1_1':keras_batch}, batch_size=1, verbose=1)

print "MLP: ", np.argmax(keras_features)
