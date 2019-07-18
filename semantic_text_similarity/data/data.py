from pkg_resources import resource_exists, resource_listdir, resource_string, resource_stream,resource_filename

def load_sts_b_data():
    """
    Loads the STS-B dataset found here: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark

    :return a tuple containing the train, dev, test split given by the data aggregators.
    """

    if not resource_exists('semantic_text_similarity', 'data/sts_b/sts-train.csv'):
        raise FileNotFoundError('Cannot find STS-B dataset')

    train = resource_string('semantic_text_similarity', 'data/sts_b/sts-train.csv').decode('utf-8').strip()
    dev = resource_string('semantic_text_similarity', 'data/sts_b/sts-dev.csv').decode('utf-8').strip()
    test = resource_string('semantic_text_similarity', 'data/sts_b/sts-dev.csv').decode('utf-8').strip()

    def yielder():
        for partition in (train, dev, test):
            data = []
            for idx,line in enumerate(partition.split('\n')):
                line = tuple(line.split("\t"))
                data.append({
                    'index': idx,
                    'sentence_1': line[5],
                    'sentence_2': line[6],
                    'similarity': float(line[4])
                })
            yield data

    return tuple([dataset for dataset in yielder()])