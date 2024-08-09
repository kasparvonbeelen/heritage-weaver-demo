import pandas as pd
import numpy as np
import scipy.spatial as sp
from .embedding_tools import ChromaDB
from matplotlib import pyplot as plt
from PIL import Image
import requests
from typing import Dict,  List, Tuple, Union, Literal

def retrieve_records(db: ChromaDB,coll: str,modality: str) -> ChromaDB:   
    filters = {
        "$and": [
            {
                "modality": {
                    "$eq": modality
                }
            },
            {
                "collection": {
                    "$eq" : coll
                }
            }
            ]
        }

    return db.get(
                where=filters,
                include=['embeddings','metadatas'] 
                    )


def get_data(db: ChromaDB,
             coll1: str, coll2: str, 
             modality1: str, 
             modality2: str) -> Dict[str, Union[str, List[str], np.ndarray]]: 
    data1 = retrieve_records(db,coll1, modality1)
    data2 = retrieve_records(db,coll2, modality2)
    print(len(data1),len(data2))
    
    inputs = dict()
    
    for i,(name,data) in enumerate([(coll1, data1), (coll2,data2)]):
        i+=1
        inputs[f'coll{i}_ids'] = data['ids']
        inputs[f'coll{i}_rids'] = [record['record_id'] for record in data['metadatas']]
        inputs[f'coll{i}_emb'] = np.matrix(data['embeddings'])
        inputs[f'coll{i}_len'] = len(data['ids'])
        inputs[f'coll{i}_name'] = name
    
    return inputs

def compute_similarities(inputs: Dict[str, Union[str, List[str], np.ndarray]],
                                      agg_function: Literal['mean','max'],
                                      percentile: bool, 
                                      threshold: float, 
                                      binarize: bool) -> Tuple[Dict[str, Union[str, List[str], np.ndarray]], np.ndarray]:
    print('--- Get similarities ---')
    similarities = 1 - sp.distance.cdist(inputs['coll1_emb'],inputs['coll2_emb'], 'cosine')
    if percentile:
        threshold = np.percentile(similarities.reshape(-1), percentile) 

    print(f'--- Using {threshold} as threshold ---')
    print('--- Aggregate similarities by record ---')
    df = pd.DataFrame(similarities, index=inputs['coll1_rids'], columns=inputs['coll2_rids']) # 

    if agg_function == 'mean':
        similarities = df.stack().reset_index().groupby(['level_0','level_1']).mean().unstack()
    elif agg_function == 'max':
        similarities = df.stack().reset_index().groupby(['level_0','level_1']).max().unstack()
    else:
        raise Exception('Aggregation function not supported, select either mean or max')
    #exec(f"similarities = df.stack().reset_index().groupby(['level_0','level_1']).{agg_function}().unstack()")
    inputs['coll1_rids'],inputs['coll2_rids'] = list(similarities.index), list(similarities.columns.droplevel())
    similarities = similarities.values

    print('--- Threshold similarities and binarize ---')
    if binarize:
        similarities[similarities >= threshold] = 1
        similarities[similarities < threshold] = 0
   
    return inputs, similarities

def get_edges(db: ChromaDB,coll1: str,coll2: str,modality1: str,modality2: str,
              agg_function: Literal['mean','max'],
              percentile: bool=False, threshold: float=.9, binarize: bool=True):
    print('Get inputs...')
    inputs = get_data(db, coll1, coll2, modality1, modality2)
    print('Compute similarities...')
    inputs, similarities = compute_similarities(inputs,agg_function, percentile, threshold, binarize)
    print('Retrieve edges...')
    mapping1 = {i:j for i,j in zip(range(len(inputs['coll1_rids'])),inputs['coll1_rids'])}
    mapping2 = {i:j for i,j in zip(range(len(inputs['coll2_rids'])),inputs['coll2_rids'])}

    edges = [(mapping1[i],mapping2[j]) for i,j in zip(*np.where(similarities > 0))]
    return edges, similarities, inputs


def plot_query_results(results: dict, 
                       source: str='img_path') -> pd.DataFrame:
    """function for plotting the results of a query"""
    result_df = pd.DataFrame(results['metadatas'][0])
    result_df['similarity'] = 1 - np.array(results['distances'][0])


    fig = plt.figure(figsize=(10, 20))
    columns = 2
    rows = 5
    for i in range(1, columns*rows +1):
        if source == 'img_url':
            try:
                img = Image.open(requests.get(result_df.loc[i-1,source], stream=True).raw).convert("RGB")
            except:
                img = Image.open('./heritageweaver/data/No_Image_Available.jpg').convert("RGB")
        elif source == 'img_path':
            img = Image.open(result_df.loc[i-1,source]).convert("RGB")
        ax = fig.add_subplot(rows, columns, i,)
        title = f"{result_df.loc[i-1,'record_id']} {result_df.loc[i-1,'similarity']:.3f}" # 
        ax.title.set_text(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.show()
    return result_df

def get_query_results(results: Dict, 
                      #collection_df: pd.DataFrame, 
                      source: str='img_path') -> pd.DataFrame:
    result_df = pd.DataFrame(results['metadatas'][0])
    result_df['similarity'] = 1 - np.array(results['distances'][0])
    #top_results = result_df.groupby('record_id')['similarity'].max().sort_values(ascending=False)#.index.tolist()
    # return pd.DataFrame(
    #             top_results
    #         ).merge(
    #             collection_df[['record_id',source,'description']],
    #             left_index=True,
    #             right_on='record_id',
    #             how='left'
    #                 ).reset_index(drop=True)
    
    return result_df