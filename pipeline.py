import luigi
import csv
import numpy as np
from ast import literal_eval
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import pickle
from shutil import copy


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file contains just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def run(self):
        with open(self.tweet_file, encoding="latin-1") as f_in:
            reader = csv.reader(f_in)
            cleaning_column = next(reader).index("tweet_coord")
        with open(self.tweet_file, encoding="latin-1") as f_in:
            with open(self.output_file, 'w', encoding="UTF-8", newline="") as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                writer.writerow(next(reader))
                for row in reader:
                    if row[cleaning_column] != '' and row[cleaning_column] != "[0.0, 0.0]":
                        writer.writerow(row)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file has columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):
        self.clean_cities()
        with open(self.input().path, encoding="UTF-8") as tweets:
            with open(self.output_file, 'w', newline="") as long_data:
                tweet_reader = csv.reader(tweets)
                tweet_header = next(tweet_reader)
                tweet_coord_column = tweet_header.index("tweet_coord")
                tweet_sentiment_column = tweet_header.index("airline_sentiment")
                writer = csv.writer(long_data)
                writer.writerow(("y", "X"))
                for tweet in tweet_reader:
                    tweet_coord = np.array(literal_eval(tweet[tweet_coord_column]), dtype='float64')
                    with open(self.cities_file, encoding="UTF-8") as cities:
                        cities_reader = csv.reader(cities, quoting=csv.QUOTE_ALL)
                        city_header = next(cities_reader)
                        coord_columns = (city_header.index("latitude"), city_header.index("longitude"))
                        city_column = city_header.index("asciiname")
                        index_1_row = next(cities_reader)
                        min_city = index_1_row[city_column]
                        min_value = np.linalg.norm(
                            np.array([index_1_row[k] for k in coord_columns], dtype='float64') - tweet_coord)
                        for row in cities_reader:
                            if min_value > np.linalg.norm(
                                    np.array([row[k] for k in coord_columns], dtype='float64') - tweet_coord):
                                min_value = np.linalg.norm(
                                    np.array([row[k] for k in coord_columns], dtype='float64') - tweet_coord)
                                min_city = row[city_column]
                        if tweet[tweet_sentiment_column] == "negative":
                            writer.writerow((0, min_city))
                        elif tweet[tweet_sentiment_column] == "neutral":
                            writer.writerow((1, min_city))
                        elif tweet[tweet_sentiment_column] == "positive":
                            writer.writerow((2, min_city))
                        else:
                            print(f"Error: unknown tweet sentiment. {tweet}")

    def output(self):
        return luigi.LocalTarget(self.output_file)

    # The cities.csv file cannot be fully read by Python csv.reader (23279 rows instead of 23775 like in notepad).
    # There are 6 pairs of instances of the csv's delimiter switching from "," to "\t" mid-line.
    # Csv.reader can import 23768 rows from the new file. The remaining 7 must be amongst the 12 where the delimiter
    # changes mid-line, and can be fixed or imported by hand. (Can easily be found using notepad++).
    def clean_cities(self):
        copy(self.cities_file, self.cities_file.split(".")[0] + "_plus_tab_delimiter.csv")
        with open(self.cities_file, encoding="UTF-8") as cities:
            cities_reader = csv.reader(cities, delimiter='\t')

            with open(self.cities_file.split(".")[0] + "_plus_tab_delimiter.csv", 'a', encoding="UTF-8", newline="") as clean:
                writer = csv.writer(clean, delimiter=',')
                next(cities_reader)
                for row in cities_reader:
                    if len(row) == 19:
                        writer.writerow(row)
        self.cities_file = self.cities_file.split(".")[0] + "_plus_tab_delimiter.csv"


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file is the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def run(self):
        long_df = pd.read_csv(self.input().path, header=0)
        binarizer = LabelBinarizer()
        logistic_model = LogisticRegression(multi_class="multinomial", solver='newton-cg')
        logistic_model.fit(binarizer.fit_transform(long_df['X']), long_df["y"])
        model_and_binarizer = {"model": logistic_model, "binarizer": binarizer}
        with open(self.output_file, 'wb') as file:
            pickle.dump(model_and_binarizer, file, pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file is a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        return TrainModelTask(self.tweet_file)

    def run(self):
        with open(self.input().path, 'rb') as file:
            model_and_binarizer = pickle.load(file)
        number_of_cities = len(model_and_binarizer["binarizer"].classes_)

        with open(self.output_file, 'w', newline="") as scores:
            writer = csv.writer(scores)
            writer.writerow(["city", "negative", "neutral", "positive"])
            for x in range(number_of_cities):
                covariate_vector = np.zeros(shape=(1, number_of_cities))
                covariate_vector[0, x] = 1
                writer.writerow([model_and_binarizer["binarizer"].classes_[x], *model_and_binarizer["model"].predict_proba(covariate_vector)[0]])
        self.sort_by_positive()     # could replace with an implement of merge sort?

    def output(self):
        return luigi.LocalTarget(self.output_file.split(".")[0] + "_sorted_positive.csv")

    def sort_by_positive(self):
        pd.read_csv(self.output_file, header=0).sort_values(by=["positive"]).to_csv(self.output_file.split(".")[0] + "_sorted_positive.csv", header=True, index=False)


if __name__ == "__main__":
    luigi.run()
