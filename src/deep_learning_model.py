from collections import Counter

from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from data_processing import load_dataset

if __name__ == '__main__':
    train, label_train = load_dataset("pkl-objects/train_titles.pkl")
    test, label_test = load_dataset("pkl-objects/test_titles.pkl")
    words = []
    for x in train:
        words.extend(x)
    vocabulary = Counter(words)
    batch_size = 100
    vocabulary_size = 100000
    # define Tokenizer with Vocab Size
    tokenizer = Tokenizer(num_words=vocabulary_size)
    train = [' '.join(row) for row in train]
    test = [' '.join(row) for row in test]

    tokenizer.fit_on_texts(train)

    x_train = tokenizer.texts_to_matrix(train, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(test, mode='tfidf')

    y_train = to_categorical(label_train, num_classes=2)
    y_test = to_categorical(label_test, num_classes=2)

    model = Sequential()
    model.add(Dense(512, input_shape=(vocabulary_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=30,
                        verbose=1,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1])
