import tensorflow as tf
import numpy as np
from tensorflow import keras

data = keras.datasets.imdb

(train_data, train_lables), (test_data, test_lables) = data.load_data(num_words=88000)

print(test_lables[0])

word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UKN>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=250, padding='post',
                                                        value=word_index['<PAD>'])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=250, padding='post', value=word_index['<PAD>'])

# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=keras.activations.relu))
# model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
#
# model.summary()
#
# model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
#
# x_train = train_data[:10000]
# x_val = train_data[10000:]
#
# y_train = train_lables[:10000]
# y_val = train_lables[10000:]
#
#
# model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val))
# model.save('imdb.h5')

model = keras.models.load_model('imdb.h5')
result = model.evaluate(test_data, test_lables)


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


with open("test_review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(encode)
        print(predict[0])


print(result)
