from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dropout
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

import sys, os
import pandas as pd
np = pd.np

from sklearn.model_selection import train_test_split
from sklearn import metrics

from yelp_analysis import Yelp

assert os.path.exists('./data/embeddings/'), 'please put word embeddings of glove/word2vec/fasttext/... \
                                                in "./data/embeddings/" folder'

class TextClassification:

    def __init__(self, EMBEDDINGS_FILE_PATH='./data/embeddings/glove.6B.100d.txt', MAX_SEQUENCE_LENGTH=1000,
                        MAX_NB_WORDS=100000, EMBEDDING_DIM=100, VALIDATION_SPLIT=0.2, **kwargs):

        # Max. word counts in a review (more data will be clipped & less will be padded)
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.MAX_NB_WORDS = MAX_NB_WORDS            # Max. Size of Vocabulary
        self.EMBEDDING_DIM = EMBEDDING_DIM          # Size of embedding dimensions (glove)
        self.VALIDATION_SPLIT = VALIDATION_SPLIT    # validation split

        self.yp = Yelp()
        self.hg = self.yp.get_holy_grail_data()

        print ('loading embeddings...')
        # load glove or any other embeddings
        self.embeddings_index = {}
        with open(os.path.join(EMBEDDINGS_FILE_PATH)) as fl:
            for line in fl:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

    def load_data(self, training_samples=100000, validation_samples=20000, testing_samples=10000):

        # total no. of reviews
        total_data = self.hg.count()

        # get the total number of samples to extract from spark DataFrame
        total_samples = training_samples + validation_samples + testing_samples
        sample_fraction = total_samples / total_data

        # Sample all classes with same rate
        sampling_fractions = dict(enumerate(np.ones(5) * sample_fraction, start=1))
        df = self.hg.sampleBy('stars', fractions=sampling_fractions, seed=7).toPandas()

        # Another way to do the sampling. But with equal number of all classes
        # # initialize an empty Dataframe
        # df = self.yp.sqlContext.createDataFrame(self.yp.sc.emptyRDD(), schema=self.hg.schema)
        #
        # # Extract a stratified sample for 5 stars (1 to 5)
        # for i in range(1, 6):
        #     df = df.union(self.hg.filter(self.hg.stars == i).limit(data_per_sample))
        #
        # df = df.toPandas()

        df.columns = ['review', 'gender', 'stars']
        df.gender = df.gender.astype(int)
        df.stars = df.stars.astype(int)

        # Remove reviews with 'NA' values
        df.dropna(inplace=True)

        # Divide the dataset into training and validation sets
        # stratify it to avoid class imbalance
        x_train, x_val, y_train, y_val = train_test_split(df.review, df.stars,
                                                            test_size=self.VALIDATION_SPLIT,
                                                            random_state=7, stratify=df.stars)

        texts = x_train.tolist() + x_val.tolist()
        labels = y_train.tolist() + y_val.tolist()

        # Tokenize and create sequences (padding & clipping of reviews)
        tokenizer = Tokenizer(nb_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        self.word_index = tokenizer.word_index

        data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

        # binarize the categorical labels
        labels = to_categorical(np.asarray(labels))[:, 1:]

        # now because of the way the 'texts' & 'labels' are appended,
        # splitting at the 'validation point' avoids class imbalance
        nb_validation_sampless = int(self.VALIDATION_SPLIT * data.shape[0])

        x = data[:-nb_validation_sampless]
        y = labels[:-nb_validation_sampless]

        x_val = data[-nb_validation_sampless:]
        y_val = labels[-nb_validation_sampless:]

        # Now that we have a good validation data, let's create a test set as well
        # Doing this later was necessary as the whole data had to be tokenized and 'sequenced'
        # before the split
        nb_test_samples = int(self.VALIDATION_SPLIT/2 * x.shape[0])

        x_train = x[:-nb_test_samples]
        y_train = y[:-nb_test_samples]

        x_test = x[-nb_test_samples:]
        y_test = y[-nb_test_samples:]

        # Sample the data to test on single machines
        self.x_train, self.x_val, self.x_test = x_train[:training_samples], x_val[:validation_samples], \
                                                x_test[:testing_samples]
        self.y_train, self.y_val, self.y_test = y_train[:training_samples], y_val[:validation_samples], \
                                                y_test[:testing_samples]

        del x_train, x_val, x_test, y_train, y_val, y_test

        return self

    def _prep_embedding_layer(self):
        nb_words = min(self.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((nb_words + 1, self.EMBEDDING_DIM))
        # First word is kind of a '<TOKEN>' used for special purposes
        # Introduce some small variance instead of pure zeroes
        embedding_matrix[0] = np.random.uniform(-0.25, 0.25, self.EMBEDDING_DIM)

        for word, i in self.word_index.items():
            if i > self.MAX_NB_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                # randomly initialize with some variance for Out-of-Vocab words
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25, self.EMBEDDING_DIM)

        # Set trainable=True, so that the weights tune to the task at hand
        # Allow dropout to regularize word embeddings as well.
        # Based on the paper: https://arxiv.org/pdf/1512.05287v5.pdf
        self.embedding_layer = Embedding(nb_words + 1,
                                            self.EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=self.MAX_SEQUENCE_LENGTH,
                                            trainable=True,
                                            dropout=0.2)

        return self

    def build_network(self):
        self._prep_embedding_layer()
        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        # Input the embedding layer at the start before everything else
        embedded_sequences = self.embedding_layer(sequence_input)

        x = Conv1D(300, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        x = Dense(150, activation='relu')(x)
        preds = Dense(5, activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                            optimizer='nadam',
                            metrics=['categorical_accuracy'])  # Or 'accuracy' can also be used

        return self

    def train(self, num_epochs=3, batch_size=500):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                        nb_epoch=num_epochs, batch_size=batch_size, callbacks=[early_stopping])

    def test(self):
        pred = self.model.predict(self.x_test)

        # convert the binarized ones to integer labels
        true = self.y_test.argmax(1)
        pred = pred.argmax(1)

        print ('categorical_accuracy: {}'.format(metrics.accuracy_score(true, pred)))
        print ('confusion_matrix:\n {}\n'.format(metrics.confusion_matrix(true, pred)))

        ar = metrics.confusion_matrix(true, pred)
        mean_accuracy = np.mean([ ar[i][i]/ar[i].sum() for i in range(5) ])

        print ('\n mean accuracy of classwise accuracies (from the above confusion_matrix): {}'.format(mean_accuracy))


if __name__ == '__main__':
    tc = TextClassification()
    tc.load_data(1000, 200, 100)
    tc.build_network()
    tc.train()
    tc.test()
