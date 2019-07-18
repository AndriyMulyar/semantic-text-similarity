# semantic-text-similarity
an easy-to-use interface to fine-tuned BERT models for computing semantic similarity. that's it.

This project contains an interface to fine-tuned, BERT-based semantic text similarity models. It slightly modifies [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-transformers) by abstracting away all the research code for ease of real-world applicability.

| Model             | Training Dataset | Dev. Correlation |
|-------------------|------------------|------------------|
| Web STS BERT      | STS-B            |                  |
| Clinical STS BERT | MED-STS          |                  |

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
```python
from semantic_text_similarity import WebSTSBert

model = WebSTSBert()

model.predict("She won an olympic gold medal","The women is an olympic champion")
model.predict("She won an olympic gold medal","The women is an olympic champion")
```
More [examples](/examples).



# Notes
- You will need a GPU to apply these models if you would like any hint of speed in your predictions.
