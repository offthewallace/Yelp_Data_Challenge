""" Code to do exploratory data analysis in Yelp """

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, DateType, ArrayType, StringType

import os, sys
import string
import pandas as pd
np = pd.np
pd.set_option('display.expand_frame_repr', False)

from matplotlib import pyplot as plt
import seaborn as sns

from utils import Utils

sc = SparkContext('local[*]')
spark = SparkSession(sc)
sqlContext = SQLContext(sc)


class Yelp:

    def __init__(self, data_folder='./data/'):
        global sc, sqlContext, spark

        self.sc = sc
        self.sqlContext = sqlContext
        self.spark = spark

        self.ut = Utils()

        # Load data
        self.users = self.spark.read.json('{}/yelp_dataset/yelp_academic_dataset_user.json'.format(data_folder))
        self.reviews = spark.read.json('{}/yelp_dataset/yelp_academic_dataset_review.json'.format(data_folder))
        self.tips = spark.read.json('{}/yelp_dataset/yelp_academic_dataset_tip.json'.format(data_folder))
        self.biz = spark.read.json('{}/yelp_dataset/yelp_academic_dataset_business.json'.format(data_folder))

        # State codes and their respective country codes
        self.states = {
                'UK'  : ('KHL', 'NTH', 'HAM', 'MLN', 'SCB', 'FIF', 'ELN', 'EDH', 'XGL'),
                'USA' : ('SC', 'NC', 'AZ', 'MN', 'AK', 'FL', 'TX', 'PA', 'NM', 'AL', 'IL', 'CA', 'WI', 'NV', 'TAM'),
                'DEU' : ('BW', 'NW', 'RP'),
                'CAN' : ('QC', 'ON'),
        }

        self.preProcess_data()

    def preProcess_data(self, names_folder='./data/names/'):
        """ Assigns gender and converts dates from string to datetime related ops on DataFrames """

        name_counts = self.ut.get_name_counts(names_folder)
        naamen = dict(name_counts.reset_index()[['name', 'gender']].values)

        def _assign_gender(name):
            split = [x.strip() for x in name.lower().replace('.', ' ').split()]
            name = split[np.argmax([len(x) for x in split])] if len(split) else ''
            return naamen.get(name, 2)

        gend_to_col = udf(_assign_gender, IntegerType())
        self.users = self.users.withColumn('gender', gend_to_col('name'))

        # store male & female user_ids
        self.mid = self.users.filter(self.users.gender == 1).select('user_id')
        self.fid = self.users.filter(self.users.gender == 0).select('user_id')

        str_to_full_datetime = udf(lambda x : pd.to_datetime(x), DateType())
        self.tips = self.tips.withColumn('date', str_to_full_datetime(col('date')))
        self.reviews = self.reviews.withColumn('date', str_to_full_datetime(col('date')))
        self.users = self.users.withColumn('yelping_since', str_to_full_datetime(col('yelping_since')))

        datetime_to_endOfMonth = udf(lambda x : x.to_period('M').to_timestamp('M').to_datetime(), DateType())
        self.tips = self.tips.withColumn('mod_date', datetime_to_endOfMonth(col('date')))
        self.reviews = self.reviews.withColumn('mod_date', datetime_to_endOfMonth(col('date')))
        self.users = self.users.withColumn('yelping_since', datetime_to_endOfMonth(col('yelping_since')))

        word_count = udf(lambda x: len(x.split()), IntegerType())
        self.tips = self.tips.withColumn('wordCount', word_count(col('text')))
        self.reviews = self.reviews.withColumn('wordCount', word_count(col('text')))

        self.tips.registerTempTable('tips')
        self.reviews.registerTempTable('reviews')
        self.users.registerTempTable('users')
        self.biz.registerTempTable('business')

        self.mf = self.users.filter(self.users.gender != 2).select('user_id', 'yelping_since', 'gender')
        self.mf.registerTempTable('maleFemale_table')


    def get_overall_growth_rate(self, source='tips'):
        """
            Get overall growth rate of 'tips'/'reviews'

            Parameters:
            ---------
                source: must be either 'tips' / 'reviews'

            Returns:
            --------
                Pandas DataFrame containing 'dates' & 'counts'
        """

        if source == 'tips':
            source_df = self.tips
        else:
            source_df = self.reviews

        source_df.registerTempTable('source_df')
        growth = spark.sql("""  SELECT mod_date AS dt, COUNT(mod_date) AS counts
                                FROM source_df
                                GROUP BY mod_date
                                ORDER BY mod_date
                            """)

        return growth.toPandas()


    def get_growth_rate(self, country='USA', source='tips', byGender=False):
        """
            Get monthly growth rates of 'reviews' / 'tips' by country & Gender (optional)

            Parameters:
            -----------
                country: string, must be one of ['USA', 'UK', 'CAN', 'DEU']
                source:  string, must be one of ['tips', 'reviews']
                byGender: boolean

            Returns:
            --------
                one or more Pandas DataFrames containing 'dates' & 'counts'
        """

        # self.sqlContext.clearCache()

        source_df = self.tips if source == 'tips' else self.reviews

        print ('\n\n obtaining growth rates for the country {}\n\n'.format(country))
        state_codes = self.states.get(country)
        print ('obtaining local business ids')
        biz_ids = self.spark.sql('SELECT business_id FROM business WHERE state IN {}'.format(state_codes))

        print('obtaining local business users')
        biz_uids = source_df.join(biz_ids, 'business_id').select('user_id').distinct()

        if byGender:
            print('getting male & female local users')
            male_uids   = biz_uids.join(self.mid, 'user_id')
            female_uids = biz_uids.join(self.fid, 'user_id')

            print ('getting male & female reviewing/tipping dates')
            male_growth   = source_df.join(male_uids,   'user_id').select('mod_date')
            female_growth = source_df.join(female_uids, 'user_id').select('mod_date')

            male_growth.registerTempTable('male_growth')
            female_growth.registerTempTable('female_growth')

            print ('getting male date counts')
            male_tip_growth = spark.sql(""" SELECT mod_date, COUNT(mod_date)
                                            FROM male_growth
                                            GROUP BY mod_date
                                            ORDER BY mod_date
                                        """)

            mtg = male_tip_growth.toPandas()
            mtg.columns = ['dt', 'm{}counts'.format(country)]

            print ('getting female date counts')
            female_tip_growth = spark.sql("""   SELECT mod_date, COUNT(mod_date)
                                                FROM female_growth
                                                GROUP BY mod_date
                                                ORDER BY mod_date
                                          """)

            ftg = female_tip_growth.toPandas()
            ftg.columns = ['dt', 'f{}counts'.format(country)]

            print ('normalizing the counts...')
            mtg['m{}counts'.format(country)] /= mtg['m{}counts'.format(country)].sum()
            ftg['f{}counts'.format(country)] /= ftg['f{}counts'.format(country)].sum()

            print ('converting to datetime...')
            mtg.dt = mtg.dt.astype(np.datetime64)
            ftg.dt = ftg.dt.astype(np.datetime64)

            mtg.set_index('dt', inplace=True)
            ftg.set_index('dt', inplace=True)

            # print ('resampling...')
            # mtg = mtg.set_index('dt').resample('1M').asfreq().fillna(0)
            # ftg = ftg.set_index('dt').resample('1M').asfreq().fillna(0)

            print ('returning male & female date growth counts')
            return mtg, ftg

        else:

            print('getting reviewing/tipping dates')
            user_dates = source_df.join(biz_uids, 'user_id').select('mod_date')
            user_dates.registerTempTable('user_dates')

            print ('getting date counts')
            user_growth = spark.sql(""" SELECT mod_date, COUNT(mod_date)
                                        FROM user_dates
                                        GROUP BY mod_date
                                        ORDER BY mod_date
                                    """)
            tg = user_growth.toPandas()
            tg.columns = ['dt', '{}counts'.format(country)]

            print ('normalizing the counts...')
            tg['{}counts'.format(country)] /= tg['{}counts'.format(country)].sum()

            print ('converting to datetime...')
            tg.dt = tg.dt.astype(np.datetime64)

            # print ('resampling...')
            # tg = tg.set_index('dt').resample('1M').asfreq().fillna(0)

            print ('returning date growth counts')
            return tg


    def plot_wordCloud(self, dataFrame, textColumnName):
        """
            Given a dataframe with a text column, writes it to disk and plots wordcloud

            Parameters:
            -----------
                dataFrame: pySpark DataFrame, source of data
                textColumnName: string, name of the text column in the dataframe to fetch text

            Returns:
            -------
                None
        """

        df = dataFrame.select(textColumnName).toPandas()
        self.ut.plot_wordCloud(' '.join(df[textColumnName].tolist()))


    def get_gender_counts(self, source='tips', per_day_or_month='day'):
        """
            Get the daily/monthly counts of users/tips/reviews

            Parameters:
            -----------
                source: string, must be one of ['tips', 'reviews', 'users']
                per_day_or_month: string, must be one of ['day', 'month']
                                    ignored, if source == 'users'

            Returns:
            --------
                Pandas DataFrame containing 'dates' & 'counts'
        """

        # self.sqlContext.clearCache()

        if source == 'users':
            source_df = self.users
        elif source == 'reviews':
            source_df = self.reviews
        elif source == 'tips':
            source_df = self.tips
        else:
            print ('unknown source. Must be one of ["tips", "reviews", "users"]')
            return

        source_df.registerTempTable('source_df')

        if source != 'users':
            date = 'date'
            if per_day_or_month == 'day':
                mf_df = self.spark.sql("""   SELECT s.date, mf.gender
                                             FROM source_df s, maleFemale_table mf
                                             WHERE mf.user_id = s.user_id
                                       """)
            else:
                mf_df = self.spark.sql("""   SELECT s.mod_date as date, mf.gender
                                             FROM source_df s, maleFemale_table mf
                                             WHERE mf.user_id = s.user_id
                                       """)
        else:
            mf_df = self.users
            date = 'yelping_since'

        mf_df.registerTempTable('mf_df')
        male_counts = self.spark.sql("""  SELECT {0} AS date, COUNT(gender) AS male_count
                                          FROM mf_df
                                          WHERE gender = 1
                                          GROUP BY {0}
                                      """.format(date))

        female_counts = self.spark.sql(""" SELECT {0} AS date, COUNT(gender) AS female_count
                                           FROM mf_df
                                           WHERE gender = 0
                                           GROUP BY {0}
                                       """.format(date))

        gender_counts = male_counts.join(female_counts, 'date')
        gender_counts.registerTempTable('gender_counts')

        gender_counts = self.spark.sql('SELECT * FROM gender_counts ORDER BY date')

        return gender_counts.toPandas()


    def get_holy_grail_data(self):
        """
            Get rich data for classification (hence holy grail). Currently only limited to US data.

            Parameters:
            -----------
                None

            Returns:
            --------
                pySpark DataFrame containing 'review', 'gender', 'stars'
        """

        # self.sqlContext.clearCache()
        usa_biz_ids = self.spark.sql(""" SELECT business_id
                                         FROM business
                                         WHERE state IN {}""".format(self.states.get('USA')))
        rv_usa = self.reviews.join(usa_biz_ids, 'business_id').select('text', 'user_id', 'stars')
        usa_genders = self.users.join(rv_usa, 'user_id').select('text', 'stars', 'gender')\
                                                        .filter(col('gender') != 2)

        hg = usa_genders.drop_duplicates(['text'])

        PUNCTUATION = str.maketrans({key: ' ' for key in string.punctuation})
        preProcess = udf(lambda x: ' '.join(x.lower().replace('\n', ' ').translate(PUNCTUATION).split()))

        self.holy_grail = hg.withColumn('review', preProcess(col('text'))).select('review', 'gender', 'stars')

        print ('holy_grail obtained...')
        return self.holy_grail
