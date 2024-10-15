from transformers import AutoProcessor, AutoModel
import chromadb
from typing import NewType
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import importlib
from chromadb.utils.data_loaders import ImageLoader
from chromadb import Documents, EmbeddingFunction, Embeddings
from PIL import JpegImagePlugin
import torch
import numpy as np
import pandas as pd
from spacy.lang.en import English 
from tqdm.notebook import tqdm
from typing import Dict,  List, Tuple, Union, Literal

nlp = English()
nlp.add_pipe('sentencizer')
#kw_model = KeyBERT()

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

class ImageLoaderRGB():
    def __init__(self, max_workers: int = multiprocessing.cpu_count()) -> None:
        try:
            self._PILImage = importlib.import_module("PIL.Image")
            self._max_workers = max_workers
        except ImportError:
            raise ValueError(
                "The PIL python package is not installed. Please install it with `pip install pillow`"
            )

    def _load_image(self, uri):
        #return np.array(self._PILImage.open(uri)) if uri is not None else None
        return np.array(self._PILImage.open(uri).convert('RGB')) if uri is not None else None

    def __call__(self, uris):
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._load_image, uris))


ChromaDB = NewType('ChromaDB', chromadb.PersistentClient)

def load_db(clien_path: str='hw',db_name: str="heritage_weaver", checkpoint: str='google/siglip-base-patch16-224') -> ChromaDB:
    siglip_embedder = SigLIPEmbedder(checkpoint)
    client = chromadb.PersistentClient(path=clien_path)
    data_loader = ImageLoader()
    collection_db = client.get_or_create_collection(name=db_name, 
                                             metadata={"hnsw:space": "cosine"},
                                             embedding_function=siglip_embedder, 
                                             data_loader=data_loader
                                            )
    return collection_db
    

def batchify(df: pd.DataFrame, batch_size: int=32) -> pd.DataFrame:
    """split dataframe into smaller batches of size batch_size
    Arguments:
        df (pd.DataFrame): dataframe to be split
        batch_size (int): size of each batch
    Returns:
        a generator of dataframes
    """
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:min(i+batch_size,len(df))]


def text_batch(batch: pd.DataFrame) -> Tuple[list, list]:
    """split the batch into content and metadatas for text indexing"""
    batch.loc[:,'modality'] = 'text'
    batch.loc[:,'sentence'] = batch.apply(lambda x: [x['name']]+[sent.text for sent in nlp(x.description).sents], axis=1)
    batch_exploded = batch.explode('sentence')
    content = [str(s) for s in batch_exploded.sentence]
    metadatas = batch_exploded[['record_id','name','sentence','img_url','img_path','modality','collection']].to_dict(orient='records')
    return content, metadatas    


def image_batch(batch: pd.DataFrame) -> Tuple[list, list]:
    """split the batch into content and metadatas for image indexing"""
    content = [str(path) for path in batch.img_path.values.tolist()]
    batch.loc[:,'modality'] = 'image'
    metadatas = batch[['record_id','name','img_url','img_path','modality','collection']].to_dict(orient='records') # 'description',
    return content, metadatas

def get_ids(collection_db, content: list) -> list:
    """get ids for the current batch of content to be indexed in the database
    """
    counter = collection_db.count()
    ids = [str(i) for i in range(counter,counter+len(content))]
    return ids

def index_data(collection_db: ChromaDB, collection_df: pd.DataFrame, batch_size: int=32) -> None:
    """index batch-wise a collection of text and images provided in a dataframe
    Arguments:
        collection_db (ChromaDB): the chromabd client database to be indexed
        collection_df (pd.DataFrame): the dataframe containing the collection to be indexed
        batch_size (int): the size of each batch
    """

    # because all rows in the dataframe contain or describe a single image
    # we iterate over all the rows in the dataframe and index them in the database
    print('indexing images...')
    batches = batchify(collection_df, batch_size=batch_size)

    for batch in tqdm(batches, total=collection_df.shape[0]//batch_size):
        image_content, image_metadatas = image_batch(batch)
        ids = get_ids(collection_db, image_content)

        collection_db.add(
            ids = ids,
            uris = image_content,
            metadatas = image_metadatas
        )
    
    # to remove duplicate descriptions for objects related to multiple images
    # we will deduplicate the dataframe by the 'record_id' column
    print('indexing text...')
    collection_df_deduplicated_text = collection_df.drop_duplicates(subset='record_id').reset_index(drop=True)
    batches = batchify(collection_df_deduplicated_text, batch_size=batch_size)

    for batch in tqdm(batches, total=collection_df_deduplicated_text.shape[0]//batch_size):
        text_content, text_metadatas = text_batch(batch)
        ids = get_ids(collection_db, text_content)
        
        collection_db.add(
            ids = ids,
            documents = text_content,
            metadatas = text_metadatas
        )
