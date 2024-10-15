from typing import *
from datasets import Dataset
from tqdm.notebook import tqdm
from pathlib import Path
#from PIL import Image
#from lxml import etree
from typing import Union
import pandas as pd
import numpy as np
import json
#from .embedding_tools import SigLIPEmbedder
#import torch
#import chromadb
import matplotlib.pyplot as plt
import requests
import random
import time
from json import JSONDecodeError


# ----------------------------------
# -------- Helper functions --------
# ----------------------------------

# def lower_case(examples, target_col: str='description') -> Dict[str, List[str]]:
#     """function for lower casing all text in a dataset column"""
#     return {'text': [t.lower() for t in examples[target_col]]}

# def open_image(record,target_col: str='img_path') -> Dict[str, Image.Image]:
#   """function for opening an image from a dataset column"""
#   return {'image' : Image.open(record[target_col])}


# def unique_substrings(substring: str, string_list: List[str]) -> bool:
#     """
#     Check if a substring appears in a list of strings.

#     :param substring: The substring to search for.
#     :param string_list: A list of strings to search in.
#     :return: False if the substring is found in any of the strings, False otherwise.
#     """
#     for string in string_list:
#         if substring in string and substring != string:
#             return False
#     return True


# ----------------------------------
# --- Generic Collection Class -----
# ----------------------------------

class MultiModalCollection(object):
    """Generic Collection class that provides most of 
    functionalities and tools we want to apply to the 
    multimodal online catalogues.
    """
    def __init__(self,df=None, img_folder: str='imgs'): # ,device: str='cpu'
        self.df = df
        self.img_folder = Path(img_folder)
        self.img_folder.mkdir(exist_ok=True)
        self.images = set(self.img_folder.glob('*.jpg'))   
        #self.device = device if torch.backends.mps.is_available() else "cpu"

    def update_images(self):
        self.images = set(self.img_folder.glob('*.jpg'))

    def load_from_csv(self, path_to_csv: str):
        self.df = pd.read_csv(path_to_csv, index_col=0)
        self.df.downloaded = self.df.img_path.apply(lambda x: False if pd.isnull(x) else  Path(x).is_file())
    
    def filter_records(self):
        """removes records without images automatically converts the updated 
        pandas dataframe to a hf dataset as well
        """
        self.df = self.df[(self.df.downloaded==True) & (~self.df['description'].isnull())].reset_index()
        self.to_dataset()

    def save_csv(self,out_path):
        self.df.to_csv(out_path)

    def to_dataset(self):
        """convert a pandas dataframe to a hugging face dataset instance"""
        self.dataset = Dataset.from_pandas(self.df)
    
    def __len__(self):
        return self.df.shape[0]
    
    def __str__(self):
        return f'< catalogue with {self.df.shape[0]} records >'


# ----------------------------------
# ----- SMG Collection Class -------
# ----------------------------------


class SMGCollection(MultiModalCollection):
    """Main object for processing data from the Science Museum Group online catalogue. 
    This is a subclass of the generic MultiModalCollection class that contains the functionalities
    that should apply to all collections.
    """

    def __init__(self, df: pd.DataFrame = pd.DataFrame(), img_folder: str='smg_imgs'): # , device: str='cpu'
        MultiModalCollection.__init__(self,df,img_folder) # ,device
        self.collection_name = 'smg'
   
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
                        columns=['record_id','name','description','category','taxonomy','img_loc'] # 'img_name','img_path','downloaded'
                        )
        
        self.df =  self.df.explode('img_loc')
        self.df['img_name'] =  self.df.img_loc.apply(lambda x: x.replace('/','|') if not x in ['','nan',np.nan] else '')
        self.df['img_path'] =  self.df.apply(lambda x: self.img_folder / x.img_name if not x['img_name']=='' else '', axis=1)
        self.df['downloaded'] =  self.df.img_path.apply(lambda x: True if x in self.images else False)

        # replace all nans with white space
        #self.df['base_url'] = 'https://coimages.sciencemuseumgroup.org.uk/'
        base_url = 'https://coimages.sciencemuseumgroup.org.uk/'
        self.df['img_url'] = self.df.apply(lambda x: f'{base_url}{x.img_loc.lower()}' if not x.img_loc in ['','nan',np.nan] else '', axis=1)
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
        # get the title under the title attribute
        title =  '; '.join([s.get('value','').strip() for s in source.get('title',[])])
        # whitespaces seems to split rows when saving csv
        title = ' '.join(title.split())
        # concatenate title and description
        description = title + '.\n\n' + description
        # get the names under the name attribute
        names =  '; '.join([s.get('value','').strip() for s in source.get('name',[])])
        # whitespaces seems to split rows when saving csv
        names = ' '.join(names.split())
        # get all the taxonmy terms
        terms =  source.get('terms',None)
        # get the first category
        first_category = source.get('categories',[])[0]['name']

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
        
        #img_loc, img_name, img_path = '', '', ''
        img_loc = []
        multimedia = source.get('multimedia',None)
        if multimedia:
            # get the medium file size
            for m in multimedia:
                img_loc.append(m['processed']['medium']['location']) #  #multimedia[0]
                # reformat image file name, so it correspond to local path
                # we in fetch_images replaced the forward slash with a |
                #img_name = img_loc.replace('/','|')
                #img_path = self.img_folder / img_name
        
        #downloaded = self.img_folder / img_loc.replace('/','|') in self.images
        
        return [record_id,names,description, first_category,taxonomy, img_loc ] # img_name, img_path, downloaded
    
    def fetch_images(self, n: int=0, record_ids=[]) -> None:
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
            url = (base_url + '/'+ loc).lower()
            img_name = loc.replace('/','|')
            request  = requests.get(url)
            if request.status_code == 200: # check if request is successful    
                with open(self.img_folder / img_name, 'wb') as f:
                    f.write(request.content)
                    time.sleep(random.uniform(.25, .25)) # randomize the requests
                    return True
            return False

        print('before downloading',len(self.images)) 
        base_url = 'https://coimages.sciencemuseumgroup.org.uk'
        self.df.downloaded = self.df.img_path.apply(lambda x: False if pd.isnull(x) else  Path(x).is_file())

        if n > 0:
            # get all the rows for images that are not downloaded yet and take a subset of `max_images` 
            
            img_locs_all = list(self.df[(self.df.downloaded==False) & (~self.df.img_loc.isin(['','nan',np.nan]))].img_loc)
        
            print('remaining images to download', len(img_locs_all))
            img_locs = img_locs_all[:n]
        
            # download the images
            # hard coded base url for getting images from the SMG group
            _ = [fetch_image(r) for r in tqdm(img_locs)]
            # get the number of downloaded images
        
        elif record_ids != None:
            img_locs = list(self.df[(self.df.downloaded==False) & \
                                        (~self.df.img_loc.isin(['','nan',np.nan])) &
                                        (self.df.record_id.isin(record_ids))].img_loc)
            print(img_locs)
            _ = [fetch_image(r) for r in tqdm(img_locs)]
        
        self.df.downloaded = self.df.img_path.apply(lambda x: False if pd.isnull(x) else Path(x).is_file())
        self.update_images()
        print('after downloading',len(self.images))


# ----------------------------------
# ----- NMS Collection Class -------
# ----------------------------------

class NMSCollection(MultiModalCollection):
    def __init__(self,df=None, img_folder='nms_imgs'): # ,device='cpu'
        MultiModalCollection.__init__(self,df,img_folder) # ,device
        self.collection_name = 'nms'

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
        print('Using the following files', list(files))
        dfs = [pd.read_csv(f) for f in files]
        col_names = []
        for df in dfs:
            col_names.append(set(df.columns))
        col_names = list(set.intersection(*col_names))
        
        self.df = pd.concat([df[col_names] for df in dfs], axis=0)
        #self.df.drop_duplicates(subset=['priref'], inplace=True) # Check what is the best identifier
        self.df.reset_index(inplace=True)

        col_mapping ={'object_number':'record_id', # Check what is the best identifier
         'object_name':'name',
         'object_category':'taxonomy',
         'administration_name':'category',
         'media.reference':'img_loc'}
        
        self.df.rename(col_mapping, axis=1, inplace=True)
        self.df['img_loc'] = self.df['img_loc'].apply(lambda x: x.split('|') if (not pd.isnull(x)) else [])
        self.df['img_loc'] = self.df['img_loc'].apply(lambda x: [i for i in x if i.startswith('PF') or i.startswith('MP')])
        self.df = self.df.explode('img_loc')
        self.df['img_name'] = self.df.img_loc.apply(lambda x: x + '.jpg' if not pd.isnull(x) else x)
        self.df['img_path'] = self.df.img_name.apply(lambda x: self.img_folder/ x if not pd.isnull(x) else x)
        self.df['downloaded'] = self.df.img_path.apply(lambda x: x.is_file() if not pd.isnull(x) else False)
        self.df['img_path'] = self.df['img_path'].apply(lambda x: str(x))
        base_url = 'https://www.nms.ac.uk/api/axiell?command=getcontent&server=Detail&value='
        #base_url = 'https://www.nms.ac.uk/search.axd?command=getcontent&server=Detail&value='
        self.df['img_url'] = self.df.apply(lambda x: f'{base_url}{x.img_loc}'.strip() if x.img_loc else '', axis=1)
        self.df = self.df[['record_id','name','description','category','taxonomy','img_loc','img_name','img_path','img_url','downloaded']]

    
    def load_from_csv(self,path_to_csv):
        self.df = pd.read_csv(path_to_csv, index_col=0)

    def fetch_images(self,from_dataframe=True,imgs_ids=None,**kwargs):
        base_url = 'https://www.nms.ac.uk/api/axiell?command=getcontent&server=Detail&value='
        #base_url = 'https://www.nms.ac.uk/search.axd?command=getcontent&server=Detail&value='
        if from_dataframe and not imgs_ids:
            imgs_ids = list(self.df[~self.df['img_loc'].isnull()]['img_loc'])
            imgs_ids = [i for e in imgs_ids for i in e.split('|') if i.startswith('PF')]
            
            
       
        for img in tqdm(imgs_ids):
            if (self.img_folder / ( img+'.jpg')).is_file(): 
                continue  
                
            url = base_url + img
            request  = requests.get(url)
            if request.status_code == 200: # check if request is successful  
            
                with open(self.img_folder / ( img+'.jpg'), 'wb') as f:
                    f.write(request.content)
                    time.sleep(random.uniform(.25, .25)) # randomize the requests



# ----------------------------------
# ----- BT Collection Class --------
# - Deprecated not updated for now -
# ----------------------------------

# Example usage loading and processing collection
# bt = BTCollection(img_folder='bt_imgs')
# bt.load_from_xml('data/bt_catalogue.xml')
# bt.save_csv('data/BT.csv')


# Example usage image scaping
# #scraping forbitted by BT, getting 403 error
# collection = BTCollection(img_folder=Path('bt_imgs'))
# collection.load_from_csv('data/BT.csv')
# collection.df.downloaded.sum()
# collection.fetch_images(n=10)


class BTCollection(MultiModalCollection):
    
    def __init__(self,df=None, img_folder='bt_imgs'): # ,device='cpu'
        MultiModalCollection.__init__(self,df,img_folder,device)
        self.collection_name = 'bt'

    def fetch_images(self, n=-1):
        def fetch_image(loc: str):   
           
            #url = base_url + '/'+ loc
            url = base_url + loc
            img_name = loc.split('/')[-1]
            print(url)
            request  = requests.get(url)
            print(request.status_code)
            if request.status_code == 200:
               
                with open(self.img_folder / img_name, 'wb') as f:
                    print(self.img_folder / img_name)
                    f.write(request.content)
                    time.sleep(random.uniform(.5, 1.5))
                    return True
            return False
        self.df.downloaded = self.df.img_path.apply(lambda x: False if pd.isnull(x) else  Path(x).is_file())
        print('before downloading',self.df.downloaded.sum()) 
        img_names = list(self.df[(~self.df.img_loc.isnull()) & (self.df.downloaded == False)].img_loc)
        #print(len(img_names))
        img_names =  [img for img in img_names if (not (self.img_folder / img).is_file()) and (len(img) > 0)][:n]
       
        #print('before downloading',len(self.images)) 
        base_url = 'http://www.digitalarchives.bt.com/CalmView/GetImage.ashx?db=Catalog&type=default&fname='
        for img in tqdm(img_names):
            fetch_image(img)

        #self.images = set(self.img_folder.glob('*.*')) 
        #print('after downloading',len(self.images))
        base_url = 'http://www.digitalarchives.bt.com/CalmView/GetImage.ashx?db=Catalog&type=default&fname='
        self.df['img_url'] = self.df.apply(lambda x: f'{base_url}{x.img_loc}' if x.img_loc else '', axis=1)
        self.df.downloaded = self.df.img_path.apply(lambda x: False if pd.isnull(x) else  Path(x).is_file())
        print('before downloading',self.df.downloaded.sum()) 
        


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
        self.df['category'] = ''
        self.df.rename({'RefNo':'record_id','Title':'names','Thumbnail':'img_loc','Description':'description'},
                       axis=1, inplace=True)

          


# ----------------------------------
# ------ VA Collection Class -------
# - Deprecated not updated for now -
# ----------------------------------

 
# Example usage loading collection 
# va = VACollection(img_folder='va_imgs')
# va.load_from_csv('data/VA.csv')

class VACollection(MultiModalCollection):
    def __init__(self,df=None, img_folder='va_imgs'): # ,device='cpu'
        MultiModalCollection.__init__(self,df,img_folder) # ,device
        self.collection_name = 'va'
        
    def parse_record(self,record):
        record_id = record['systemNumber']
        description = ' '.join([v.strip() for k,v in record.items() if 'description' in k.lower()])
        images = record['images'] 
        if images:
            img_loc = images[0]
            img_name = f'{img_loc}.jpg'
            img_path = str(Path(f'{self.img_folder}/{img_loc}.jpg')) # for now we just take the first image
        else:
            img_loc, img_name, img_path = '','',''
        donwloaded = False
        name = record['objectType']
        taxonomy = ', '.join([e['text'] for e in record['categories']])
        return [record_id,name,description,taxonomy,img_loc,img_name,img_path,donwloaded]
    
    def to_csv(self):
        data = json.load(open('data/VA.json'))
        rows = [self.parse_record(r['record']) for r in data]
        self.df = pd.DataFrame(rows, 
                               columns=['record_id','name','description','taxonomy','img_loc','img_name','img_path','downloaded'])
        self.df['downloaded'] = self.df.img_path.apply(lambda x: Path(x).is_file() if x else False)
        base_url = 'https://framemark.vam.ac.uk/collections/'
        postfix = '/full/600,/0/default.jpg'
        self.df['img_url'] = self.df.apply(lambda x: f'{base_url}{x.img_loc}{postfix}' if x.img_loc else '', axis=1)
        

        self.df.to_csv('data/VA.csv')

    def fetch_images(self):
        base_url = 'https://framemark.vam.ac.uk/collections/'
        postfix = '/full/600,/0/default.jpg'
        for i, row in tqdm(self.df.iterrows()):
            if (self.img_folder).is_file(): 
                continue

            if row.img_loc:
                url = base_url+row.img_loc+postfix
                request  = requests.get(url)
                if request.status_code == 200: # check if request is successful  
               
                    with open(self.img_folder / row.img_name, 'wb') as f:
                        f.write(request.content)
                        time.sleep(random.uniform(.25, .25)) # randomize the requests


    def fetch_records_api(self,query,page_size = 50):
        Path(f'data/va_json').mkdir(exist_ok=True)
        self.data_path = Path(f'data/va_json/VA_{query}.json')
        
        if self.data_path.is_file():
            with open(self.data_path) as in_json:
                self.json = json.load(in_json)
        else:
            self.json = []
        self.page = len(self.json)
        
        print(self.page)

        p = 1
        call = f'https://api.vam.ac.uk/v2/objects/search?q={query}&page={p}&page_size={page_size}'
        start_req = requests.get(call).json()
        record_count = start_req['info']['record_count'] 
        print(f'total records {record_count}')
        downloaded = (self.page)*page_size
        print(f'records downloaded {downloaded}')
        print(f'remaining total records to download {record_count - downloaded}')
        print(f'iterations left {((record_count - downloaded) // page_size)}')
       
        for page in range(record_count// page_size)[self.page:10000//page_size]:
                print(page)
                try:
                    req = requests.get(f'https://api.vam.ac.uk/v2/objects/search?q="{query}"&page={page}&page_size={page_size}')
                    self.json.append(req.json())
                except (JSONDecodeError) as e:
                    with open(self.data_path,'w') as out_json:
                        json.dump(self.json,out_json)
            
        with open(self.data_path,'w') as out_json:
            json.dump(self.json,out_json)

        
    def fetch_records(self):
        va_files = list(Path('data/va_json').glob('*.json'))
        print(len(va_files))
        records_ids = []
        for vf in va_files:
            with open(vf) as in_json:
                data = json.load(in_json)
                if len(data) > 0:
                    records_ids.extend([e['systemNumber'] for e in data[0]['records']])

        records_ids = list(set(records_ids))
        self.records = []
        print(len(records_ids))
        for rid in tqdm(records_ids):
            #print(rid)
            self.records.append(requests.get(f'https://api.vam.ac.uk/v2/museumobject/{rid}').json())
        with open('data/VA.json','w') as out_json:
            json.dump(self.records,out_json)


        