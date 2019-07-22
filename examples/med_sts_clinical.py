from semantic_text_similarity.models import ClinicalBertSimilarity
from scipy.stats import pearsonr

model = ClinicalBertSimilarity()
predictions = model.predict([("The patient is sick.", "Grass is green."),
                             ("A prescription of acetaminophen 325 mg was given."," The patient was given Tylenol.")])

print(predictions)