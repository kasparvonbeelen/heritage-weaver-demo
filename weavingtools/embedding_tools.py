from transformers import AutoProcessor, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings
from PIL import JpegImagePlugin
import torch
import numpy as np
from spacy.lang.en import English 
from tqdm.notebook import tqdm
from keybert import KeyBERT

nlp = English()
nlp.add_pipe('sentencizer')
kw_model = KeyBERT()

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
    

def batchify(content_list, n=20):
    for i in range(0, len(content_list), n):
        yield np.array(content_list[i:min(i+n,len(content_list))])

def reshape_text_batch(batch, collection):
    
    content = [[b[0],b[1],s,b[3]] 
                       for b in list(batch) 
                           for s in nlp(str(b[1])+ '. ' + str(b[2])).sents if len(s) > 2 # str(b[1])+ ', '+
                                  ] # parameter here
    
    metadatas = [{'record_id':e[0],
                 'name':e[1],
                 #'img_path': str(e[4]),
                 'img_url': str(e[3]),
                 'input_modality': 'text',
                 'sentence': str(e[2]),
                 'collection': collection
                        }
                    for e in content
                    ]
    content = [str(c[2]) for c in content]
    return content,metadatas



def reshape_image_batch(batch, collection):
    
    content = [str(i) for i in list(batch[:,-1])]
    
    metadatas = [{'record_id':e[0],
                  'name':e[1],
                  #'img_path': str(e[4]),
                  'img_url': str(e[3]),
                  'input_modality':'image',
                  'collection': collection
                        }
                    for e in batch
                    ]
    
    return content,metadatas