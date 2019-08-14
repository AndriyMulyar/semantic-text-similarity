from .similarity import BertSimilarity
from ..util import get_model_path

class ClinicalBertSimilarity(BertSimilarity):
    def __init__(self, device='cuda', batch_size=10, model_name="clinical-bert-similarity"):
        model_path = get_model_path(model_name)
        super().__init__(device=device, batch_size=batch_size, bert_model_path=model_path)