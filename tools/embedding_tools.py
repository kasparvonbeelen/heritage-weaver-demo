from transformers import AutoProcessor, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings
from PIL import JpegImagePlugin
import torch
import numpy as np

class SigLIPEmbedder(EmbeddingFunction):
    """embedding function for indexing (batches of) text or images"""
    def __init__(self, checkpoint: str):
        self.model = model = AutoModel.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.checkpoint = checkpoint 

    def __str__(self):
        return f'< {self.checkpoint} Embedding >'
        
    def __call__(self, input: Documents) -> Embeddings:
        # all inputs are either images or texts
        # since they are passed as either uris or text arguments
        # when creating the vector database
        with torch.no_grad():
            if isinstance(input[0],(JpegImagePlugin.JpegImageFile, np.ndarray)):
           
                #images = [Image.open(p) for p in input]
                inputs = self.processor(images=input, return_tensors="pt",padding='max_length')# padding="max_length")
                outputs = self.model.vision_model(**inputs)

            elif isinstance(input[0],str):
                inputs = self.processor(text=input, return_tensors="pt", padding='max_length',truncation=True)
                outputs = self.model.text_model(**inputs)

            else:
                raise Exception('Input should either be a path to an image file or a string')

        # taken from original code in transformers library
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L1095
        embeddings = outputs[1]
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        
        return embeddings.tolist()