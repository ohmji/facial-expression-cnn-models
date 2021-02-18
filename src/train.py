import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, Xception, VGG16, InceptionV3
from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout, \
    Flatten, Dense, BatchNormalization, \
    SpatialDropout2D, AveragePooling2D, Input, GlobalAveragePooling2D, Activation, ZeroPadding2D, Convolution2D
import os
import cv2
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn import model_selection

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.get_logger().setLevel('WARNING')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', type=str, default='data/raw_dataset',
                    help="Directory of dataset")
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help="Where to write the new data")
parser.add_argument("-m", "--model", type=str, default="face_emotion.model",
                    help="Path to output face mask detector model")
parser.add_argument('-s', '--size', type=int, default=48,
                    help="Size of input data")
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help="Bactch size of data generator")
parser.add_argument('-l', '--learning-rate', type=float, default=0.0001,
                    help="Learning rate value")
parser.add_argument('-sh', '--show-history', action='store_true',
                    help="Show training history")
parser.add_argument('-n', '--net-type', type=str, default='MobileNetV2',
                    choices=['CNN', 'MobileNetV2', 'VGG16', 'Xception'],
                    help="The network architecture, optional: CNN, MobileNetV2, VGG16, Xception")


def CNN_model(learning_rate, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer=Adam(learning_rate=learning_rate))
    return model


def MobileNetV2_model(learning_rate, input_shape):
    baseModel = MobileNetV2(
        include_top=False, input_tensor=Input(shape=input_shape),
        # weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        # weights='imagenet'
    )
    model = Sequential()
    model.add(baseModel)
    # for layer in baseModel.layers[:-4]:
    #     layer.trainable = False

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer=Adam(learning_rate=learning_rate))
    return model


def VGG16_model(learning_rate, input_shape):
    baseModel = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      include_top=False,)
    model = Sequential()
    model.add(baseModel)
    for layer in baseModel.layers[:-7]:
        layer.trainable = False

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer=Adam(learning_rate=learning_rate))
    return model


def Xception_model(learning_rate, input_shape):
    baseModel = Xception(
        include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer=Adam(learning_rate=learning_rate))
    return model


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    Ref: https://stackoverflow.com/a/64359137
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p)
                           for p in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(p)
                               for p in model.non_trainable_weights])

    total_memory = (batch_size * shapes_mem_count + internal_model_mem_count
                    + trainable_count + non_trainable_count)
    return total_memory


def preprocess_data():
    data = pd.read_csv('fer2013.csv')
    labels = pd.read_csv('fer2013new.csv')

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48
    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 3))
    for i in range(n_samples):
        image = np.fromstring(data['pixels'][i], dtype=int,
                              sep=' ').reshape((h, w, 1))
        # last_axis = -1
        # grscale_img_3dims = np.expand_dims(image, last_axis)
        # dim_to_repeat = 2
        # repeats = 3
        # X[i] = np.repeat(grscale_img_3dims, repeats, dim_to_repeat)
        X[i] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
        # print(X[i])

    return X, y


def clean_data_and_normalize(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen


def save_model_and_weights(model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    model_json = model.to_json()
    with open('Saved-Models/model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize and save weights to JSON
    model.save_weights('Saved-Models/model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')


if __name__ == "__main__":

    args = parser.parse_args()

    bs = 64
    lr = args.learning_rate
    size = (48, 48)
    shape = (48, 48, 3)
    epochs = 20

    # # Load and preprocess data
    # train_dir = os.path.join(args.data_dir, 'train')
    # test_dir = os.path.join(args.data_dir, 'test')
    # valid_dir = os.path.join(args.data_dir, 'validation')

    # train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range=0.2,
    #                                    shear_range=0.2, brightness_range=[0.9, 1.1],
    #                                    horizontal_flip=True)
    # valid_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range=0.2,
    #                                    shear_range=0.2, brightness_range=[0.9, 1.1],
    #                                    horizontal_flip=True)
    # test_datagen = ImageDataGenerator(rescale=1./255)

    # train_generator = train_datagen.flow_from_directory(train_dir, target_size=size, shuffle=True,
    #                                                     batch_size=bs, class_mode='binary')
    # valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=size, shuffle=True,
    #                                                     batch_size=bs, class_mode='binary')
    # test_generator = test_datagen.flow_from_directory(test_dir, target_size=size, shuffle=True,
    #                                                   batch_size=bs, class_mode='binary')

    # print(train_generator.class_indices)
    # print(train_generator.image_shape)

    X, y = preprocess_data()
    X, y = clean_data_and_normalize(X, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    datagen = data_augmentation(x_train)

    # Build model
    net_type_to_model = {
        'CNN': CNN_model,
        'MobileNetV2': MobileNetV2_model,
        'VGG16': VGG16_model,
        'Xception': Xception_model
    }
    model_name = args.net_type
    model_builder = net_type_to_model.get(model_name)
    model = model_builder(lr, shape)
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + f"-size-{size[0]}" +
                                              f"-bs-{bs}" + f"-lr-{lr}.h5"),
                                 monitor='val_loss', save_best_only=True, verbose=1)
    # Train model
    # history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator,
    #                     batch_size=bs, callbacks=[earlystop, tensorboard, checkpoint], shuffle=True)

    history = model.fit(datagen.flow(x_train, y_train, batch_size=bs), epochs=epochs,
                        steps_per_epoch=len(x_train) // bs,
                        batch_size=bs, callbacks=[
        earlystop, tensorboard, checkpoint], shuffle=True,
        validation_data=(x_val, y_val), verbose=2)

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, batch_size=bs)
    metrics = pd.DataFrame(history.history)
    print(metrics.head(10))

    print('test_loss: ', test_loss)
    print('test_accuracy: ', test_accuracy)
    print('Memory consumption: %s bytes' %
          keras_model_memory_usage_in_bytes(model, batch_size=bs))

    # serialize the model to disk
    print("saving mask detector model...")

    # save_model_and_weights(model, test_accuracy)
    model.save(model_name+'_'+args.model, save_format="h5")

    if args.show_history:
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()

        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()
