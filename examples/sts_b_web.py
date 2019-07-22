from semantic_text_similarity.data import load_sts_b_data
from semantic_text_similarity import WebBertSimilarity
from scipy.stats import pearsonr

train, dev, test = load_sts_b_data()

model = WebBertSimilarity()
predictions = model.predict(dev)


print(pearsonr([instance["similarity"] for instance in dev], predictions))




