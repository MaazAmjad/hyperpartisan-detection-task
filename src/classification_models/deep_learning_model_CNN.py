import numpy as np
from keras.models import Model

from preprocessing.data_processing import load_dataset

if __name__ == '__main__':
    train, label_train = load_dataset("pkl-objects/train_titles.pkl")
    test, label_test = load_dataset("pkl-objects/test_titles.pkl")

    # Constants
    batch_size = 1000
    vocabulary_size = 1000
    maxlen = 20

    print("betch_size: {}, vocabulary_size: {}, length_of_sequence: {}".format(batch_size, vocabulary_size, maxlen))
    # Data pre processing
    x_train = np.array(train)
    x_test = np.array(test)

    print(x_train.shape)
    # Labels
    y_train = label_train
    y_test = label_test

    model = Model

    model.summary()
    # Train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=5,
                        verbose=1,
                        validation_split=0.1)

    # Evaluate model on testing data
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    model.save('model_with_embeddings.h5')

    print('Test accuracy:', score[1])
