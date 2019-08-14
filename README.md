# semantic-text-similarity
an easy-to-use interface to fine-tuned BERT models for computing semantic similarity. that's it.

This project contains an interface to fine-tuned, BERT-based semantic text similarity models. It modifies [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) by abstracting away all the research benchmarking code for ease of real-world applicability.

| Model             |          Dataset | Dev. Correlation |
|-------------------|------------------|------------------|
| Web STS BERT      | STS-B            |     0.893        |
| Clinical STS BERT | MED-STS          |     0.854        |

# Installation

Install with pip:

```
pip install semantic-text-similarity
```

or directly:

```
pip install git+https://github.com/AndriyMulyar/semantic-text-similarity
```

# Use
Maps batches of sentence pairs to real-valued scores in the range [0,5]
```python
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction

clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction

web_model.predict([("She won an olympic gold medal","The women is an olympic champion")])
```
More [examples](/examples).



# Notes
- You will need a GPU to apply these models if you would like any hint of speed in your predictions.
- Model downloads are cached in `~/.cache/torch/semantic_text_similarity/`. Try clearing this folder if you have issues.
