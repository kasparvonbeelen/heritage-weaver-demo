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

def filter_edges_by_keywords(edges: List[Tuple[str,str]], 
                             db: ChromaDB, 
                             keywords: List[str],
                             max_candidates: int=1000,) -> List[Tuple[str,str]]:
    """function to filter edges by keywords
    Arguments:
        edges: list, list of edges
        db: ChromaDB object
        keywords: list, list of keywords to filter the edges"""
    
    # get the metadata for the records
    results = db.query(
                query_texts=keywords,
                #include=['record_id','description'],
                n_results=max_candidates
                    )
    edge_ids = [r['record_id'] for res in results['metadatas'] for r in res]
    # filter the edges if contains edge_ids from the results
    filtered_edges = [edge for edge in edges if edge[0] in edge_ids or edge[1] in edge_ids]
    return filtered_edges

def get_data(db: ChromaDB,
             coll1: str, coll2: str, 
             modality1: str, 
             modality2: str) -> Dict[str, Union[str, List[str], np.ndarray]]: 
    """function to get data from two different collection from the vector databases
    Arguments:
        db: ChromaDB object
        coll1: str, name of the first collection
        coll2: str, name of the second collection
        modality1: str, modality of the first collection
        modality2: str, modality of the second collection
    Returns:
        inputs: dict, dictionary containing the following keys with {i} being either 1 or 2:
            coll{i}_ids: list, list of ids from the database 
            coll{i}_rids: list, list of record ids from the database
            coll{i}_emb: np.ndarray, matrix of embeddings
            coll{i}_len: int, length of the ids
            coll{i}_name: str, name of the collection"""
    
    data1 = retrieve_records(db,coll1, modality1)
    data2 = retrieve_records(db,coll2, modality2)
    
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
                                      percentile: Union[bool,float], 
                                      threshold: float, 
                                      binarize: bool) -> Tuple[Dict[str, Union[str, List[str], np.ndarray]], np.ndarray]:
    """function to compute similarities between records from two collections
    takes as input the dictioanry produced by the get_data function
    Arguments:
        agg_function: str, aggregation function to use to aggregate the similarities by record
        percentile: bool, if False, used the value for threshold argument
            if a float, compute this threshold as the percentile cutoff of the similarities
        threshold: float, value to use as threshold if percentile is False
        binarize: bool, if True, binarize the similarities into edges
    """

    print('--- Get similarities ---')
    # get coisine similarity between embeddings matrices
    similarities = 1 - sp.distance.cdist(inputs['coll1_emb'],inputs['coll2_emb'], 'cosine')

    # we need to select a cutoff threshold to binarize the similarities into edges 
    # if percentile is True, compute the threshold as the percentile of the similarities
    if percentile:
        threshold = np.percentile(similarities.reshape(-1), percentile) 

    print(f'--- Using {threshold} as threshold ---')
    print('--- Aggregate similarities by record ---')
    df = pd.DataFrame(similarities, index=inputs['coll1_rids'], columns=inputs['coll2_rids']) # 

    # to compute the similarity by object we need to aggragate the similarities by record
    # we can either take the mean or the max of the similarities
    if agg_function == 'mean':
        similarities = df.stack().reset_index().groupby(['level_0','level_1']).mean().unstack()
    elif agg_function == 'max':
        similarities = df.stack().reset_index().groupby(['level_0','level_1']).max().unstack()
    else:
        raise Exception('Aggregation function not supported, select either mean or max')
    
    inputs['coll1_rids'],inputs['coll2_rids'] = list(similarities.index), list(similarities.columns.droplevel())
    similarities = similarities.values

    if binarize:
        print('--- Threshold similarities and binarize ---')
        similarities[similarities >= threshold] = 1
        similarities[similarities < threshold] = 0
   
    return inputs, similarities

def get_edges(db: ChromaDB,coll1: str,coll2: str,modality1: str,modality2: str,
              agg_function: Literal['mean','max'],
              percentile: Union[bool,float]=False, 
              threshold: float=.9, 
              binarize: bool=True):
    """function to get edges from two collections builds 
    on the compute_similarities and get_datafunction
    Arguments:
        db: ChromaDB object
        coll1: str, name of the first collection
        coll2: str, name of the second collection
        modality1: str, modality of the first collection
        modality2: str, modality of the second collection
        agg_function: str, aggregation function to use to aggregate the similarities by record
        percentile: bool, if False, used the value for threshold argument
            if a float, compute this threshold as the percentile cutoff of the similarities
        threshold: float, value to use as threshold if percentile is False
        binarize: bool, if True, binarize the similarities into edges"""
    
    print('Get inputs...')
    inputs = get_data(db, coll1, coll2, modality1, modality2)
    print('Compute similarities...')
    inputs, similarities = compute_similarities(inputs,agg_function, percentile, threshold, binarize)
    print('Retrieve edges...')
    # we need to map the indices to the record ids
    mapping1 = {i:j for i,j in zip(range(len(inputs['coll1_rids'])),inputs['coll1_rids'])}
    mapping2 = {i:j for i,j in zip(range(len(inputs['coll2_rids'])),inputs['coll2_rids'])}
    # get the edges from the similarities matrix 
    # assumuses a binarized matrix with edges as non-zero values
    edges = [(mapping1[i],mapping2[j]) for i,j in zip(*np.where(similarities > 0))]
    return edges, similarities, inputs


def plot_query_results(results: Dict, 
                       source :Literal['img_path', 'img_url'] = 'img_path') -> pd.DataFrame:
    """function for plotting the results of a query
    Arguments:
        results: dict, dictionary containing the results of the query
        source: str, source of the images, either 'img_path' or 'img_url'"""
    result_df = pd.DataFrame(results['metadatas'][0])
    result_df['similarity'] = 1 - np.array(results['distances'][0])


    fig = plt.figure(figsize=(10, 20))
    columns = 2
    rows = 5
    for i in range(1, columns*rows +1):
        if source == 'img_url':
            try:
                img = Image.open(requests.get(img_path,  stream=True).raw,).convert('RGB')
            except:
                try:
                    img_path = 'https://www.nms.ac.uk/api/axiell?command=getcontent&server=Detail&value=' + img_path.split('value=')[-1]
                    data = requests.get(img_path)
                    img = Image.open(io.BytesIO(bytes(data.content)))
                except:
                    #img = Image.open('./heritageweaver/data/No_Image_Available.jpg').convert("RGB")
                    no_image_uri = 'https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg?20200913095930'
                    img = Image.open(requests.get(no_image_uri,  stream=True).raw,).convert('RGB')
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
                      collection_df: pd.DataFrame, 
                      source: Literal['img_path', 'img_url'] = 'img_path'
                      ) -> pd.DataFrame:
    """function to get the results of a query
    Arguments:
        results: dict, dictionary containing the results of the query
        collection_df: pd.DataFrame, dataframe containing the collection metadata
        source: str, source of the images, either 'img_path' or 'img_url'"""
    result_df = pd.DataFrame(results['metadatas'][0])
    result_df['similarity'] = 1 - np.array(results['distances'][0])
    #result_df = result_df[['record_id',source,'modality','name','similarity']]
    top_results = result_df.groupby('record_id')['similarity'].max().sort_values(ascending=False)#.index.tolist()
    return pd.DataFrame(
                top_results
            ).merge(
                collection_df[['record_id',source,'description']],
                left_index=True,
                right_on='record_id',
                how='left'
                    ).reset_index(drop=True)
    
    