from unittest import TestCase
from semantic_text_similarity.data import load_sts_b_data
class TestData(TestCase):

    def test_load_train_dev_test_sts_b(self):
        train, dev, test = load_sts_b_data()
        return len(train) == 5749 and len(dev) == 1500 and len(test) == 1379