from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, concatenate
from keras.models import Model


# activation functions
activation = 'relu'
last_activation = "elu"

# VGG model
def vggmodel():

    model = VGG16(weights='imagenet', include_top=False)

    # choose layers for training
    for layer in model.layers[:18]:
        layer.trainable = False
    for layer in model.layers[18:]:
        layer.trainable = True
    return model


def get_eye_tracker_model(img_cols, img_rows, img_ch):

    #right eye model
    right_eye_input = Input(shape=(img_cols, img_rows, img_ch))
    vggModel = vggmodel()
    vggModel.name = 'Right_eye'
    right_eye_net = vggModel(right_eye_input)


    # left eye model
    left_eye_input = Input(shape=(img_cols, img_rows, img_ch))
    vggModel1 = vggmodel()
    vggModel1.name = 'Left_eye'
    left_eye_net = vggModel1(left_eye_input)

    # face model
    face_input = Input(shape=(img_cols, img_rows, img_ch))
    vggModel2 = vggmodel()
    vggModel2.name = 'face'
    face_net = vggModel2(face_input)

    # face grid
    face_grid = Input(shape=(1, 25, 25))

    # dense layers for eyes
    e = concatenate([right_eye_net, left_eye_net])
    e = Flatten()(e)
    fc_e1 = Dense(2048,activation=activation)(e)

    #dense layers for face

    f = Flatten()(face_net)
    fc_f1 = Dense(2048, activation=activation)(f)
    # fc_f2 = Dense(1024, activation=activation)(fc_f1)

    # dense layers for face grid
    fg = Flatten()(face_grid)
    fc_fg1 = Dense(4096, activation=activation)(fg)
    # fc_fg2 = Dense(2048, activation=activation)(fc_fg1)

    # concatenate face and face grid
    fc_fg = concatenate([fc_f1, fc_fg1])
    fc_fcfg1 = Dense(2048, activation=activation)(fc_fg)

    # combining all FC layers
    h = concatenate([fc_e1, fc_fcfg1])
    fc1 = Dense(1024, activation=activation)(h)
    fc2 = Dense(2, activation=last_activation)(fc1)

    # final model
    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])
    return final_model