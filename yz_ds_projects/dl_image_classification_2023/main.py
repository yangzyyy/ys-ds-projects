import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


def get_dataset(data_dir, classes, is_training=False):
    """The function generates normalized and augmented training dataset or normalized testing dataset.

    Arguments:
        data_dir (str):    The directory of the dataset.
        classes (dict):          The dictionary of category label.
        is_training (bool):    True if input is training dataset directory and False if is testing dataset directory.

    Return:
        train_accuracy (float): The training accuracy.
    """
    if is_training:
        # Define the image data generator for training data
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     width_shift_range=0.1,  # randomly shift images horizontally by up to 10%
                                     height_shift_range=0.1,  # randomly shift images vertically by up to 10%
                                     zoom_range=0.1,  # randomly zoom images by up to 10%
                                     horizontal_flip=True,  # randomly flip images horizontally
                                     )
    else:
        # Define the image data generator for testing data
        datagen = ImageDataGenerator(rescale=1. / 255)

    # Generate the dataset from the images in the directory
    dataset = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
        seed=42
    )
    return dataset


def basic_model():
    """The function returns basic CNN model.

    Arguments: None

    Return:
        model: The basic model architecture.
    """
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.001),
               input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dropout(0.15),
        Dense(15, activation='softmax')
    ])

    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def vgg_model():
    """The function returns tuned VGG16 model.

    Arguments: None

    Return:
        model: The model architecture ready to be trained.
    """
    # Load the pre-trained VGG16 model and freeze its layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add a new fully connected layer and a softmax output layer
    x = Flatten()(vgg16.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(15, activation='softmax')(x)

    # Define a new model that includes the VGG16 model and the new layers
    model = Model(inputs=vgg16.input, outputs=predictions)

    for layer in model.layers[:-2]:
        layer.trainable = False

    # Compile the model with a categorical cross-entropy loss and an Adam optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    return model


def scheduler(epoch, lr):
    """The function updates learning rate of the optimizer.

    Arguments:
        epoch (int):    The integer representing the current epoch number.
        lr (float):   The current learning rate.

    Return:
        updated lr (float): The updated learning rate.
    """
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def get_labels(dataset):
    """The function returns labels from input dataset.

    Arguments:
        dataset (dataset):    The dataset that contains each image's information and its label.

    Return:
        labels (list): the list of labels from dataset.
    """
    labels = []
    for i in range(int(dataset.samples / 32) + 1):
        labels.extend(list(np.argmax(dataset[i][1], axis=1)))
    labels = np.array(labels)

    return labels


def get_accuracy(y, y_pred):
    """The function calculates accuracy of predicted labels.

    Arguments:
        y (list):    The list of true labels.
        y_pred (list):   The list of predicted labels.

    Return:
        accuracy (float): The prediction accuracy.
    """
    acc = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            acc = acc + 1
    accuracy = acc / len(y)
    return accuracy


def train(train_data_dir, trained_cnn_dir, classes, model_select):
    """Main training model.

    Arguments:
        train_data_dir (str):    The directory of the training dataset.
        trained_cnn_dir (str):   The directory to save the trained model.
        classes (dict):          The dictionary of category label.
        model_select (str):      The string representing model selection. One is 'vgg', and the other is 'basic'.

    Return:
        train_accuracy (float): The training accuracy.
    """
    train_dataset = get_dataset(train_data_dir, classes, is_training=True)
    if model_select == 'vgg':
        model = vgg_model()
    elif model_select == 'basic':
        model = basic_model()
    lr_scheduler = LearningRateScheduler(scheduler)
    history = model.fit(train_dataset,
                        epochs=30,
                        steps_per_epoch=len(train_dataset),
                        verbose=1,
                        callbacks=[lr_scheduler]
                       )

    # Save model as .pkl
    model.save(trained_cnn_dir)

    # calculate accuracy
    y_train = get_labels(train_dataset)
    y_train_predict = np.argmax(model.predict(train_dataset), axis=1)

    accuracy = get_accuracy(y_train, y_train_predict)

    return accuracy


def test(test_data_dir, model_dir, classes):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The directory of testing dataset.
        model_dir (str):        The directory of the saved model.
        classes (dict):         The dictionary of category label.

    Return:
        test_accuracy (float): The testing accuracy.
    """

    model = tf.keras.models.load_model(model_dir)

    test_dataset = get_dataset(test_data_dir, classes, is_training=False)

    # calculate accuracy
    y_test = get_labels(test_dataset)
    y_test_predict = np.argmax(model.predict(test_dataset), axis=1)

    accuracy = get_accuracy(y_test, y_test_predict)

    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train', 'test'])
    parser.add_argument('--model_select', default='vgg', choices=['basic', 'vgg'])
    parser.add_argument('--train_data_dir', default='./train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./model.pkl', help='the pre-trained model')
    opt = parser.parse_args()

    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    '''
       The vgg trained model is already saved in the model_dir, if you do not want to train the model again,
       you could set phase to 'test' and run the code.
       '''

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir, labels, opt.model_select)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir, labels)
        print(testing_accuracy)



