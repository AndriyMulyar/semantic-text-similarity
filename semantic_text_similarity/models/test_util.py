from unittest import TestCase
from .util import get_model_path
import os

class TestSimilarityModelUtilities(TestCase):
    def test_model_download(self):
        path = get_model_path('web-bert-similarity')
        self.assertTrue(os.path.exists(path))