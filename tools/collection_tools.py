from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer #, util
from transformers import  AutoModel, AutoFeatureExtractor #, AutoTokenizer
from tensorboard.plugins import projector
#from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from lxml import etree
from typing import Union
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
import PIL
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import requests
import random
import time



# ----------------------------------
# -------- Helper functions --------
# ----------------------------------

def lower_case(examples, target_col='description'):
    return {'text': [t.lower() for t in examples[target_col]]}

def open_image(record,target_col='img_path'):
  return {'image' : Image.open(record[target_col])}

# def cls_pooling(model_output):
#     return model_output.last_hidden_state[:, 0]

# ----------------------------------
# -- Congruence Collection Class ---
# ----------------------------------

class CongruenceCollection(object):
    """
    meta class that can combine various instances of the Collection class
    and combines them in preparation for cross-collection analysis
    """
    def __init__(self,collection_dict: dict, only_images: bool=False):
        """Initialize an instance containining multiple collections

        Arguments:
            collection_dict (dict): a dictionary that maps collection names
                to an instance of the Collection class
            only_images (bool): keep only observations for which we 
                    have downloaded an image file
        """
        self.collection_dict = collection_dict
        self.combine_datasets(only_images)

    def combine_datasets(self, only_images: bool=False):
        """function to glue different instancesn fo the Collection class
        important note, we only retain columns that appear in all collections

        Argument:
            only_images (bool): keep only observations for which we 
                    have downloaded an image file
        """

        # get all the column names in each collection
        # and then only retain to the intersection 
        # to build the metacollection
        col_names = []
        for name, coll in self.collection_dict.items():
            col_names.append(set(coll.dataset.column_names))
        col_names = set.intersection(*col_names)
        
        # concatente datasets
        for name, coll in self.collection_dict.items():
            self.collection_dict[name].dataset = coll.dataset.map(lambda x: {'collection_name':name})
        self.dataset = concatenate_datasets(list([v.dataset.remove_columns([c for c in v.dataset.column_names
                                                                                if c not in col_names])
                                                     for v in self.collection_dict.values()]))
        
        # optional, filter based on the presence of an image
        if only_images:
            self.dataset = self.dataset.filter(lambda x: x['downloaded']==True)
        
        # convert dataset to pandas dataframe
        self.df = self.dataset.to_pandas()

    def prepare_projector(self,embedding_name,log_dir='log/congr'):
        """prepare data for embedding projector"""
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.make_sprite()
        self.write_metadata()

        # Save embeddings in TF variable.
        embeddings_vectors = self.dataset[embedding_name]
        emb = tf.Variable(embeddings_vectors, name=embedding_name)

        # Add ops to save and restore all the variables.
        checkpoint = tf.train.Checkpoint(embedding=emb)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        embedding.sprite.image_path = 'sprite.jpg' # Specify the width and height of a single thumbnail. 
        width_to_height_ratio = 1
        embedding.sprite.single_image_dim.extend([int(100*width_to_height_ratio), 100]) 
        return config
        
    def make_sprite(self, unit_size=100):
        """make sprite images for embedding projector"""
        one_square_size = int(np.ceil(np.sqrt(len(self.dataset))))
        master_width = unit_size * one_square_size
        master_height = unit_size * one_square_size
        spriteimage = Image.new(
            mode='RGBA',
            size=(master_width, master_height),
            color=(0,0,0,0) # fully transparent
        )
        for i, row in self.df.iterrows():
            div, mod = divmod(i, one_square_size)
            h_loc = unit_size * div
            w_loc = unit_size * mod
            image = Image.open(row.img_path).resize((unit_size,unit_size))
            #image.thumbnail((100,100))
            spriteimage.paste(image, (w_loc, h_loc))
        spriteimage.convert('RGB').save(f'{self.log_dir}/sprite.jpg', transparency=0)

    def write_metadata(self):
        labels = list(self.df.collection_name.values)
        with open(f'{self.log_dir}/metadata.tsv', 'w') as file: 
            for label in labels:
                file.write(f'{label}\n')

# ----------------------------------
# --- Generic Collection Class -----
# ----------------------------------

class Collection(object):
    """Generic Collection class that provides most of 
    functionalities and tools we want to apply to all 
    online catalogues.
    """
    def __init__(self,df=None, img_folder: str='imgs',device: str='cpu'):
        self.df = df
        self.img_folder = Path(img_folder)
        self.img_folder.mkdir(exist_ok=True)
        self.images = set(self.img_folder.glob('*.jpg'))   
        self.indices = dict()
        self.device = device if torch.backends.mps.is_available() else "cpu"

    def load_from_csv(self, path_to_csv: str):
        self.df = pd.read_csv(path_to_csv, index_col=0)
    
    def save_csv(self,out_path):
        self.df.to_csv(out_path)

    def to_dataset(self):
        """convert a pandas dataframe to a hugging face dataset instance"""
        self.dataset = Dataset.from_pandas(self.df)
    
    def __len__(self):
        return self.df.shape[0]
    
    def __str__(self):
        return f'< SGM catalogue with {self.df.shape[0]} records >'
    
    def split_by_sentence(self, target_col: str, min_length: int=5):
        """function that rearranges the collection by sentence
        it changes the dataframe (or df) attribute, each row represents
        now one sentence in the target_col (instead of an object).
        target_col is usually the description.

        Arguments:
            target_col (str): determines which column will be split
                into sentences and used rearrange the dataframe
            
            min_length (int): optionally, you can remove very short sentences
                this will remove any sentences with fewer than 
                min_length characters
        """

        # use the spacy pipeline but only the
        # the sentencizer, not the complete set of tools
        from spacy.lang.en import English 
        nlp = English()
        nlp.add_pipe('sentencizer')

        def get_sents(cell_input) -> list:
            """helper function for converting a cell to a string
            
            Arguments:
                cell_input: technically a cell from the dataframe
                    that we want to split into sentences
                
            Returns:
                a list containing sentences
            """
            doc = nlp(str(cell_input))
            return list(doc.sents)

        # add sentence-split text fiels as a new column
        self.df[f'{target_col}_sentences'] = self.df[target_col].apply(get_sents)
        # explode the dataframe, each row represents now one sentence
        self.df = self.df.explode(f'{target_col}_sentences')
        # now we filter to remove very short sentences
        # we keep track of the number of observations  
        # we removed in this process
        rows_before_filtering = self.df.shape[0]
        self.df = self.df[
                    self.df.description_sentences.apply(len) >= min_length 
                    ].reset_index(drop=True)
        self.df[f'{target_col}_sentences']= self.df[f'{target_col}_sentences'].astype(str)
        print(f'Filtering removed {rows_before_filtering - self.df.shape[0]} rows.')
    
    def encode_text(self,examples, target_col: str):
        """convert text field to embedding"""
        return {f'{target_col}_embedding': self.model.encode(examples[target_col])}
    
    def load_text_model(self,model_ckpt: str):
        """load model for converting text to embeddings"""
        self.model = SentenceTransformer(model_ckpt)
        self.model.to(self.device)
    
    def embed_text(self,target_col,model_ckpt):
        """main function for embedding text"""
        
        # convert to hugging face dataset
        if not hasattr(self,'dataset'): 
            self.to_dataset()

        # load hugging face model
        if not hasattr(self,'model'):
            self.load_text_model(model_ckpt)

        # remove empty rows with no text
        self.dataset = self.dataset.filter(
                            lambda x: x[target_col] != None
                                )
        
        # lowercase text
        self.dataset = self.dataset.map(
                            lower_case, batched=True,fn_kwargs={'target_col':target_col}
                                )
        
        # convert lowercased text to an embedding
        self.dataset= self.dataset.map(
                            self.encode_text, fn_kwargs={'target_col':target_col}
                                )
    
    def load_img_model(self, img_model_ckpt: str):
        """load model for converting images to an embedding"""
        # load image feature extractor
        self.img_extractor = AutoFeatureExtractor.from_pretrained(img_model_ckpt)

        # load image model
        self.img_model = AutoModel.from_pretrained(img_model_ckpt)
        self.hidden_dim = self.img_model.config.hidden_size

        # move model to device if GPU is available
        self.img_model.to(self.device)

    def set_transformation_chain(self):
        """set transformation chain for transforming images
        before features extraction. this code is copied from the 
        hugging face tutorial see
        https://huggingface.co/blog/image-similarity
        """
        self.transformation_chain = T.Compose(
            [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * self.img_extractor.size["height"])),
        T.CenterCrop(self.img_extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=self.img_extractor.image_mean, std=self.img_extractor.image_std),
            ]
        )

    def embed_query_image(self,image: PIL.Image): # TO DO: check type
        """create an embedding for the query image in the same
        way as the existing images are embedded

        Arguments:
            image (PIL.Image)
        """
        batch = {'pixel_values': torch.stack([self.transformation_chain(image)])}
        with torch.no_grad():
            query_embedding = self.img_model(**batch).last_hidden_state[:, 0].detach().cpu().numpy()[0]
        return query_embedding

    def extract_img_embeddings(self,batch: dict):  # TO DO: check type
        images = batch['image']
        # `transformation_chain` is a compostion of preprocessing
        # transformations we apply to the input images to prepare them
        # for the model. For more details, check out the accompanying Colab Notebook.
        image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed}
        with torch.no_grad():
            embeddings = self.img_model(**new_batch).last_hidden_state[:, 0].detach().cpu().numpy()
        return {f"image_embedding": embeddings}

    def embed_image(self,
                    target_col: str, 
                    model_ckpt: str="google/vit-base-patch16-224", 
                    batch_size: int=24):
        """Obtain image features

        Arguments:
            target_col (str): defines which column is used 
                    to take and embed images
            model_ckpt (str): model type and location
            batch_size (int): number of images to process per batch 
        """

        # check if hugging face dataset already exist
        if not hasattr(self,'dataset'):
            self.to_dataset()

        # check if we loaded the model
        if not hasattr(self,'img_model'):
            self.load_img_model(model_ckpt)
        
        # open images
        self.dataset = self.dataset.map(open_image, fn_kwargs={'target_col': target_col})

        # filter retain only RGB images
        # !! IMPORTTANT !! this is a small bug, and we need to probably fix this later
        # by converting grayscale to RGB
        # TO DO: make this fitlering step redundant by ensuring all images are in RGB format
        self.dataset = self.dataset.filter(lambda x: x['image'].mode=='RGB')

        # initiate transformation chain
        self.set_transformation_chain()
        # convert images and save in dataset
        self.dataset = self.dataset.map(self.extract_img_embeddings, batched=True, batch_size=batch_size)

    def extract_clip_embedding(self,record: dict) -> dict:
        """create clip embedding"""
        return {'clip_embedding':self.clip_model.encode(record['image'])#.detach().cpu().numpy()
                }

    def load_clip_model(self,clip_model_ckpt: str='clip-ViT-B-32'):
        """load clip model and convert with sentence transformer"""
        self.clip_model = SentenceTransformer(clip_model_ckpt)
        self.clip_model.to(self.device)

    def embed_clip(self,target_col,model_ckpt: str="openai/clip-vit-base-patch32"):
        """embed and image with clip"""
        if not hasattr(self,'clip_model'):
            self.load_clip_model(model_ckpt)

        self.dataset = self.dataset.map(open_image, fn_kwargs={'target_col': target_col})
        self.dataset = self.dataset.map(self.extract_clip_embedding)      


    def build_faiss_index(self, target_col: str, index_name: str):
        """build a faiss index for querying by vector 
        based on information in the target_col"""
        self.indices[index_name] = self.dataset.add_faiss_index(column=target_col)

    def query_collection(self,query ,  # Union[str, PIL.Image]
                        field: str, 
                        index_type: str ,
                        k: int=10) -> pd.DataFrame: # TO DO check type
        """query a faiss index given an query
        a query can be either on image or an text string
        additionally, we need to specify what type of the faiss index 
        we want to query as CLIP can take both image and text as input

        Arguments:
            query Union[str, PIL.Image]: the query can be either an
                a string or an image
            field (str): defines the dataset field which we want to query
            index_type (str): defines the faiss index we want to query
            k (int): number of query results we want to return
        """

        # convert the query to a query vector
        if index_type == 'text':
            query_vector = self.model.encode(query)
        elif index_type == 'image':
            query_vector = self.embed_query_image(query)
        elif index_type =='clip':
            query_vector = self.clip_model.encode(query)

        # get n nearest examples to the query vector
        scores, samples = self.indices[index_type].get_nearest_examples(
                            field, 
                            query_vector, 
                            k=k
                            )

        # convert results to a dataframe and return this dataframe
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        # smaller scores indicate higher similarity
        # we thus sort results in ascending order
        samples_df.sort_values("scores", ascending=True, inplace=True)
        samples_df.reset_index(drop=True, inplace=True)
        return samples_df
    
    def load_faiss_index(self,path_to_faiss: str, 
                        embedding_col: str,
                        field: str):
        """load existing faiss index"""
        self.to_dataset()
        self.dataset.load_faiss_index(embedding_col,path_to_faiss)
        self.indices[field] = self.dataset
    
    def plot_images(self, query_df):
        """plot nearest neighbour images"""
        fig = plt.figure(figsize=(20, 6))
        columns = 7
        rows = 3
        for i in range(1, columns*rows +1):
            
            img = query_df.loc[i-1,'image']
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()


# ----------------------------------
# ----- SMG Collection Class -------
# ----------------------------------


class SMGCollection(Collection):
    """Main object for processing data from the
    Science Museum Group online catalogue. 
    This is a subclass of the generic Collection
    class that contains most of the functionality
    that should apply to all collections.
    """

    def __init__(self, df: pd.DataFrame = pd.DataFrame(), img_folder: str='imgs', device: str='cpu'):
        self.df = df
        self.img_folder = Path(img_folder)
        self.img_folder.mkdir(exist_ok=True)
        self.images = set(self.img_folder.glob('*.jpg'))   
        self.indices = dict()
        self.device = device if torch.backends.mps.is_available() else "cpu"
   
    def load_from_json(self, path_to_json: str):
        """load the collection from a json file which
        contains an output of the original database

        Argument:
            path_to_json (str): path to the json dump
        """
        data = []
        with open(path_to_json,'r') as in_json:
            # iterate over all the dictionaries
            for d in tqdm(in_json):
                raw_record = json.loads(d) 
                processed_record = self.process_json_record(raw_record)
                data.append(processed_record)
        self.df = pd.DataFrame(
                        data, 
                        columns=['record_id','name','description','taxonomy','img_loc','img_name','img_path','downloaded']
                        )
        # replace all nans with white space
        self.df.fillna('', inplace=True)

    def process_json_record(self,record: dict) -> list:
        """Principal function for processing records in the json dump
        of the smg collection. Here we collection the most important pieces
        of information we want to use later for our experiments (either
        the multimodal analysis of )

        Arguments:
            record (dict)

        Returns:
            list with the following elements
                record_id (str): the original id as recorded by the SMG database
                names (str): names of the object, concatenated with a semicolon
                description (str): all descriptions, concatenated with a semicolon
                taxonomy (str): taxonomy terms as string but sorted according to the hierarchy
                img_loc (str): location of the medium sized image
                img_name (str): formated the name of the images
                img_path (str): path to the images
                downloaded (bool): flag indicating whether we downloaded the image
        """
        
        record_id = record['_id']
        source =  record['_source'] # get the source element
        # get all the description under the description attribute
        description = '; '.join([s.get('value','').strip() for s in source.get('description',[])])
        # whitespaces seems to split rows when saving csv
        description = ' '.join(description.split()) 
        # get all the the names under the name attribute
        names =  '; '.join([s.get('value','').strip() for s in source.get('name',[])])
        # whitespaces seems to split rows when saving csv
        names = ' '.join(description.split())
        # get all the taxonmy terms
        terms =  source.get('terms',None)
        taxonomy = ''
        if terms:
            # map all the taxonomy from sort order in the hierarchy to their name
            taxonomy = {
                        t['sort']: t['name'][0]['value'] 
                            for t in terms[0].get('hierarchy',[])
                                }
            # convert all taxonomy terms to a string in sorted order
            taxonomy = '; '.join([v.strip() for k,v in sorted(taxonomy.items()) 
                                        #if not v.startswith('<') # optional, skip terms starting with <
                                                ])
        
        img_loc, img_name, img_path = '', '', ''
        multimedia = source.get('multimedia',None)
        if multimedia:
            # get the medium file size
            img_loc =  multimedia[0]['processed']['medium']['location']
            # reformat image file name, so it correspond to local path
            # we in fetch_images replaced the forward slash with a |
            img_name = img_loc.replace('/','|')
            img_path = self.img_folder / img_name
        
        downloaded = img_path in self.images
        
        return [record_id,names,description, taxonomy, img_loc ,img_name, img_path, downloaded]
    
    
  
    def fetch_images(self, max_images: int=100) -> None:
        """Given a json dump with all records fetch
        and save images in a img_folder

        Arguments:
            max_images (int): number of images to download
        """

        def fetch_image(loc: str) -> bool:   
            """
            scrape an image by name as provided in the json file

            Arguments:
                loc (str): name of the image
            """
            url = base_url + '/'+ loc
            img_name = loc.replace('/','|')
            request  = requests.get(url)
            
            if request.status_code == 200: # check if request is successful    
                with open(self.img_folder / img_name, 'wb') as f:
                    f.write(request.content)
                    time.sleep(random.uniform(.25, .25)) # randomize the requests
                    return True
            return False

        print('before downloading',len(self.images)) 
        # get all the rows for images that are not downloaded yet and take a subset of `max_images` 
        img_locs_all = list(self.df[(self.df.downloaded==False) & (self.df.img_loc!= '')].img_loc)
        print('remaining images to download', len(img_locs_all))
        img_locs = img_locs_all[:max_images]
        
        # download the images
        # hard coded base url for getting images from the SMG group
        base_url = 'https://coimages.sciencemuseumgroup.org.uk/images'
        _ = [fetch_image(r) for r in tqdm(img_locs)]
        # get the number of downloaded images
        self.images = set(self.img_folder.glob('*.*')) 
        print('after downloading',len(self.images))
  

# ----------------------------------
# ----- BT Collection Class -------
# ----------------------------------

class BTCollection(Collection):
    
    def __init__(self,df=None, img_folder='imgs',device='cpu'):
        Collection.__init__(self,df,img_folder,device)

    def fetch_images(self, n=-1):
        def fetch_image(loc: str):   
            url = base_url + '/'+ loc
            img_name = loc.split('/')[-1]
            request  = requests.get(url)
            if request.status_code == 200:
                
                with open(self.img_folder / img_name, 'wb') as f:
                    f.write(request.content)
                    time.sleep(random.uniform(.5, 1.5))
                    return True
            return False
        
        img_names = list(self.df[~self.df.Thumbnail.isnull()].Thumbnail)
        img_names =  [img for img in img_names if not (self.img_folder / img).is_file()][:n]
 
        base_url = 'http://www.digitalarchives.bt.com/CalmView/GetImage.ashx?db=Catalog&type=default&fname='
        for img in tqdm(img_names):
            fetch_image(img)


    def load_from_xml(self,path):
        def find_and_get_text(record, element):
            result = record.find(element)
            if result is not None:
                return result.text
            return ''
        
        with open(path, 'rb') as in_xml:
            tree = etree.parse(in_xml)
        records = tree.xpath('//DScribeRecord')
        data = []
        columns = ['RefNo','Title','Thumbnail','Description']
        for r in tqdm(records):
            data.append([find_and_get_text(r,c) for c in columns])
        
        self.df = pd.DataFrame(data, columns=columns)
        self.df['img_path'] = self.df.Thumbnail.apply(lambda x: self.img_folder / x if x else x)
        self.df['downloaded'] = self.df.img_path.apply(lambda x: Path(x).is_file() if x else False)
          



# ----------------------------------
# ----- NMS Collection Class -------
# ----------------------------------

class NMSCollection(Collection):
    def __init__(self,df=None, img_folder='nms_imgs',device='cpu'):
        Collection.__init__(self,df,img_folder,device)

    def load_original_csvs(self,files):
        """Read the original CVS files containing the NMS collection
        Turns this into a combined csv with the following fields

                record_id (str): the original id as recorded by the SMG database
                names (str): names of the object, concatenated with a semicolon
                description (str): all descriptions, concatenated with a semicolon
                taxonomy (str): taxonomy terms as string but sorted according to the hierarchy
                img_loc (str): location of the medium sized image
                img_name (str): formated the name of the images
                img_path (str): path to the images
                downloaded (bool): flag indicating whether we downloaded the image

        Arguments:
            files (list): list of csv files with database exports from
                the NMS collection
        """
        dfs = [pd.read_csv(f) for f in files]
        col_names = []
        for df in dfs:
            col_names.append(set(df.columns))
        col_names = set.intersection(*col_names)
        
        self.df = pd.concat([df[col_names] for df in dfs], axis=0)
        self.df.drop_duplicates(subset=['priref'], inplace=True)
        self.df.reset_index(inplace=True)


        #{'record_id','name','description','taxonomy','img_loc','img_name','img_path','downloaded'}
    
    def load_from_csv(self,path_to_csv):
        self.df = pd.read_csv(path_to_csv, index_col=0)

    def fetch_images(self):
        imgs_ids = list(self.df[~self.df['reproduction.reference'].isnull()]['reproduction.reference'])
        imgs_ids = [i for e in imgs_ids for i in e.split('|') if i.startswith('PF')]
        base_url = 'https://www.nms.ac.uk/search.axd?command=getcontent&server=Detail&value='
        for img in tqdm(imgs_ids):
            if (self.img_folder / ( img+'.jpg')).is_file(): 
                continue  
            
            url = base_url + img
            request  = requests.get(url)
            if request.status_code == 200: # check if request is successful  
               
                with open(self.img_folder / ( img+'.jpg'), 'wb') as f:
                    f.write(request.content)
                    time.sleep(random.uniform(.25, .25)) # randomize the requests
                    