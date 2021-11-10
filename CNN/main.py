import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
for dirname, _, filenames in os.walk("/folder_location"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        np.unique(train)
        train.head()
        X = train.drop(['label'], axis=1)
        Y = train['label']

        X = X / 255.0

        test = test / 255.0
        X = X.values
        test = test.values
        X = X.reshape(-1, 28, 28, 1)


        test = test.reshape(-1, 28, 28, 1)
        Y = to_categorical(Y, num_classes=10)
        print(X.shape, Y.shape)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=20)
        print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)


        def create_model():
            model = Sequential()

            model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(28, 28, 1)
                             ))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.4))

            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.4))



            model.add(Flatten())

            model.add(Dense(64, activation="relu"))

            model.add(Dropout(0.4))

            model.add(Dense(10, activation="softmax"))

            return model


        model_CNN = create_model()

        data_aug = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)
        generator_train = data_aug.flow(X_train, Y_train, batch_size=64)
        steps = int(X_train.shape[0] / 64)
        optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=0.0000001, decay=0.0, centered=False)
        model_CNN.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint("", monitor='val_accuracy', verbose=1, save_best_only=True)
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00005, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, mode='auto',
                                   restore_best_weights=True)

        history = model_CNN.fit(generator_train, steps_per_epoch=steps, batch_size=64, epochs=50,
                                validation_data=(X_val, Y_val), verbose=1,
                                callbacks=[checkpoint, reduce_learning_rate, early_stop])



        # predict on validation set
        Y_predict_val = model_CNN.predict(X_val)

        # check the class predicted with the highest probability
        Y_pred_hp = np.argmax(Y_predict_val, axis=1)

        # check the groudtruth most common class
        Y_test_hp = np.argmax(Y_val, axis=1)

        # compare them
        accuracy_on_val = np.mean(Y_pred_hp == Y_test_hp)

        # print the accuracy
        print("Validation accuracy (after the training): ", accuracy_on_val, "\n")

        # plot the validation and training accuracy
        fig, axis = plt.subplots(1, 2, figsize=(16, 6))
        axis[0].plot(history.history['val_accuracy'], label='validation_accuracy')
        axis[0].set_title("Validation Accuracy")
        axis[0].set_xlabel("Epochs")
        axis[1].plot(history.history['accuracy'], label='accuracy')
        axis[1].set_title("Training Accuracy")
        axis[1].set_xlabel("Epochs")
        plt.show()

        # plot the Confusion Matrix
        fig, ax = plt.subplots(figsize=(12, 12))
        cm = confusion_matrix(Y_test_hp, Y_pred_hp, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        disp = disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.show()
        y_pred_test = model_CNN.predict(test)
        prediction = np.argmax(y_pred_test, axis=1)
        submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': list(prediction)})
        submission.head()
        submission.to_csv("submission.csv", index=False)
        model_CNN.save('my_model.h5')
        break


    break









