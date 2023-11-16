import os
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from dotenv import dotenv_values

env = dotenv_values(".env")
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']


class BERTNER():
    def __init__(self, model = "dslim/bert-large-NER"):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, text):
        return self.nlp(text)


        