from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from data_processing import load_dataset

if __name__ == '__main__':
    train, label_train = load_dataset("pkl-objects/train_titles.pkl")
    test, label_test = load_dataset("pkl-objects/test_titles.pkl")

    # Constants
    batch_size = 10000
    vocabulary_size = 1000
    maxlen = 100

    print("betch_size: {}, vocabulary_size: {}, length_of_sequence: {}".format(batch_size, vocabulary_size, maxlen))
    # Data pre processing
    train_x = [' '.join(row) for row in train]
    test_x = [' '.join(row) for row in test]

    # Define Tokenizer with Vocab Size
    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="UNK")
    tokenizer.fit_on_texts(train_x)

    # Convert texts to sequences
    token_tr_X = tokenizer.texts_to_sequences(train)
    token_te_X = tokenizer.texts_to_sequences(test)

    # Make sure sequences have the same length
    x_train = sequence.pad_sequences(token_tr_X, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(token_te_X, maxlen=maxlen, padding='post')

    # convert labels to 1 hot encoded vectors
    y_train = to_categorical(label_train, num_classes=2)
    y_test = to_categorical(label_test, num_classes=2)

    # Create model (1 Embedding layer, 1 Bidirectional LSTM layer, 1 Dense Layer)
    model = Sequential()
    model.add(Embedding(vocabulary_size, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile('adam',
                  'binary_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=10,
                        verbose=1,
                        validation_split=0.1)

    # Evaluate model on testing data
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1])
