import similarity
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.jarowinkler import JaroWinkler
from similarity.metric_lcs import MetricLCS
from similarity.qgram import QGram
from similarity.jaccard import Jaccard
from similarity.cosine import Cosine
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch

normalized_levenshtein = NormalizedLevenshtein()
jarowinkler = JaroWinkler()
metric_lcs = MetricLCS()
qgram2 = QGram(2)
qgram3 = QGram(3)
qgram4 = QGram(4)
cosine = Cosine(2)
jaccard = Jaccard(2)


def extract_string_similarity_vector(instance: dict):
    """
    Returns a vector encoding a variety of lexical similarity metrics given a dictionary containing keys
    sentence_1,sentence_2
    :return: a vector containing similarity scores
    """

    s1 = instance['sentence_1']
    s2 = instance['sentence_2']

    return torch.tensor([
        normalized_levenshtein.similarity(s1,s2),
        jarowinkler.similarity(s1,s2),
        metric_lcs.distance(s1,s2),
        qgram2.distance(s1,s2),
        qgram3.distance(s1,s2),
        qgram4.distance(s1,s2),
        jaccard.similarity(s1,s2),
        cosine.similarity(s1,s2),
        fuzz.partial_token_set_ratio(s1,s2),
        fuzz.partial_token_sort_ratio(s1,s2),
        fuzz.token_set_ratio(s1,s2),
        fuzz.token_sort_ratio(s1,s2),
        fuzz.QRatio(s1,s2),
        fuzz.UQRatio(s1,s2),
        fuzz.UWRatio(s1,s2),
        fuzz.WRatio(s1,s2)
    ])

def string_similarity_features(data: list):

    temp = extract_string_similarity_vector(data[0])

    dataset = torch.empty((len(data), temp.shape[0]), dtype=torch.float)

    for idx, instance in enumerate(data):
        dataset[idx] = extract_string_similarity_vector(instance)

    return dataset