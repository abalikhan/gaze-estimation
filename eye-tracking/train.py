import os
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from load_data import load_batch, load_data_names, load_batch_from_names_random
from model_vgg import get_eye_tracker_model
from keras.models import load_model


# generator for data loaded from the npz file
def generator_npz(data, batch_size, img_ch, img_cols, img_rows):

    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_ch, img_cols, img_rows)
            yield x, y


# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_cols, img_rows, img_ch):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_cols, img_rows, img_ch)
        yield x, y


# generator with random batch load (validation)
def generator_val_data(names, path, batch_size, img_cols, img_rows, img_ch):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_cols, img_rows, img_ch)
        yield x, y


def train(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    #todo: manage parameters in main

    if args.data == "big":
        dataset_path = "../data"

    if args.data == 'small':
        dataset_path = r'D:\gazecapture_small'

    if args.data == "big":
        train_path = "../dataset/train"
        val_path = "../dataset/validation"
        test_path = "../dataset/test"

    if args.data == 'small':
        train_path = r'C:\Users\Aliab\PycharmProjects\data_small/train'
        val_path = r'C:\Users\Aliab\PycharmProjects\data_small\validation'
        test_path = r'C:\Users\Aliab\PycharmProjects\data\test'

    print("{} dataset: {}".format(args.data, dataset_path))

    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    patience = args.patience

    # image parameter
    img_cols = 128
    img_rows = 128
    img_ch = 3

    # model
    model = get_eye_tracker_model(img_cols, img_rows, img_ch)

    # model summary
    model.summary()

    # weights
    # print("Loading weights...",  end='')
    # model.load_weights('weight.hdf5')
    # print("Done.")

    # optimizer
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.6, nesterov=True)
    adam = Adam(lr=1e-3)

    model_file = 'weight_vg.hdf5'
    if os.path.isfile(model_file):
        print('model loaded successfully')
        model = load_model(model_file)
        model.load_weights(model_file)
    # compile model

    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    # reduce learning rate
    red_lr = ReduceLROnPlateau(factor=0.6, monitor='val_loss', patience=2, verbose=1, min_lr=0)

    # data
    # todo: parameters not hardocoded
        # train data
    train_names = load_data_names(train_path)
        # validation data
    val_names = load_data_names(val_path)
        # test data
    test_names = load_data_names(test_path)

    # debug
    # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
    # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)


    model.fit_generator(
        generator=generator_train_data(train_names, dataset_path, batch_size, img_cols, img_rows, img_ch),
        steps_per_epoch=(len(train_names)) / batch_size,
        epochs=n_epoch,
        verbose=1,
        validation_data=generator_val_data(val_names, dataset_path, batch_size, img_cols, img_rows, img_ch),
        validation_steps=(len(val_names)) / batch_size,
        callbacks=[EarlyStopping(patience=patience), red_lr,
                    ModelCheckpoint("weight_vgg.hdf5", monitor='val_loss', save_best_only=True, verbose=1)
                    ]
        )