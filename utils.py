import pandas as pd
np = pd.np

from wordcloud import WordCloud
from glob import glob

from matplotlib import pyplot as plt


class Utils:

    def merge_text(self, folder, file_extension='txt'):
        """ Merges many files with similar extension and returns as one single large file """

        assert file_extension.strip('.') in ['txt', 'csv'], 'only "txt", "csv" extensions are supoorted Currently'

        text_files = glob('{}/*.{}'.format(folder, file_extension.strip('.')))
        dataframes = (pd.read_csv(textfile, header=None) for textfile in text_files)
        df = pd.concat(dataframes, ignore_index=True)

        return df


    def get_name_counts(self, names_folder):
        """ Reads names file and returns a DataFrame with name and its counts """

        # assign integers for genders (useful for computations)
        dc = {'m_count':1, 'f_count':0}

        # read the data
        nm = self.merge_text(names_folder, 'txt')
        nm.columns = ['name', 'gender', 'total']

        # group the data by name and aggregate based on the gender and the total counts
        grp = nm.groupby(['name', 'gender'])
        res = grp.aggregate({'total':np.sum})

        # Unstack the MultiIndex DataFrame
        name_counts = pd.DataFrame(res.unstack().reset_index().values,
                                    columns=['name', 'f_count', 'm_count'])
        name_counts.name = name_counts.name.str.lower()

        # set the total counts to zero for either male/female counts.
        # For e.g., 'Fred' might never have been assigned to a Female.
        # In that case, assign 0 to female column (default is np.NaN after groupBy)
        name_counts = name_counts.set_index('name').fillna(0)
        name_counts.m_count = name_counts.m_count.astype(int)
        name_counts.f_count = name_counts.f_count.astype(int)

        # Now assign that gender to the user that has most counts.
        name_counts['gender'] = name_counts.idxmax(axis=1).map(dc)

        return name_counts  # We obtain around 95,000 unique entries


    def align_timeseries_data(self, dfs):
        """ Chronologically align the data based on Datetime """

        mn, mx = [], []

        # Find the start and end of period
        for df in dfs:
            mn.append(df.index.min())
            mx.append(df.index.max())

        # Create empty dataframe with monthly intervals of the above period and fill in the values
        dtIndex = pd.DatetimeIndex(start=min(mn), end=max(mx), freq='M')
        df = pd.DataFrame(index=dtIndex)
        df.index.name = 'months'

        # join all the dataframes based on DatetimeIndex
        for dataframe in dfs:
            df = df.join(dataframe)

        df.fillna(0, inplace=True)

        return df


    def plot_wordCloud(self, text):
        wordcloud = WordCloud(max_font_size=60).generate(text)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()



# *** ---- do 'pip install gensim' before uncommenting the below line ---- ***
# from gensim.models import Word2Vec

# Word2Vec for Yelp
class readFile():
    """ Reads 'holy_grail_data' """
    def __init__(self, text_series):
        self.text_series = text_series

    # iteratively read just the review
    def __iter__(self):
        for line in self.text_series.iteritems():
            yield line[1].rstrip().split()


def get_model(text_series, size=300, window=5, min_count=5, workers=7,
                negative=5, sg=1, bigrams=False, **kwargs):
    """
        Generate Word2Vec model from the Yelp reviews

        Parameters:
        -----------
            text_series: pandas Series or list of sentences
            size: int, dimensionality of word embeddings to generate
            window: int, window size
            min_count: int, minimum term frequency of a given word in the corpus
            workers: int, no. of processes
            negative: int, negative sampling rate
            sg: boolean, whether to use skig-gram or not. (default:False, hence uses hierarchichal softmax)
            bigrams: boolean, whether to consider bigrams as well for deriving word vectors (default: False)

        Returns:
        --------
            Word2Vec model
    """

    sentences = readFile(text_series)
    if bigrams:
        # log.info('finding bi-grams...')
        bi_grams = Phrases(sentences)

        # log.info('training...')
        model = Word2Vec(bi_grams[sentences], size=size, window=window, min_count=min_count,
                            workers=workers, negative=negative, sg=sg, **kwargs)
    else:
        # log.info('training...')
        model = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers,
                            negative=negative, sg=sg, **kwargs)

    return model


# def save_model(model, model_name = './data/w2v_model.bin', vocab_file = './data/w2v_vocab.txt'):
#     model.save_word2vec_format(model_name, binary=True, fvocab=vocab_file)
