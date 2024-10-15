from typing import List, Tuple
from PIL import Image
import io
import matplotlib.pyplot as plt
import requests

def soft_wrap_text(text: str, max_char_length: int=40) -> str:
    words = text.split()
    lines = []
    current_line = ''
    
    for w in words:
        if len(current_line) + len(w) <= max_char_length:
            current_line+= ' ' + w if current_line else w
        else:
            lines.append(current_line)
            current_line = w
    
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def open_image(img_uri: str) -> Image.Image:
    if img_uri.startswith('http'):
        try:
            
            img = Image.open(requests.get(img_uri,  stream=True).raw,).convert('RGB')
        except:
            try:
                
                img_uri = 'https://www.nms.ac.uk/api/axiell?command=getcontent&server=Detail&value=' + img_uri.split('value=')[-1]
                data = requests.get(img_uri)
                img = Image.open(io.BytesIO(bytes(data.content)))
            except:
                no_image_uri = 'https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg?20200913095930'
                img = Image.open(requests.get(no_image_uri,  stream=True).raw,).convert('RGB')
                #img = Image.open('./data/No_Image_Available.jpg').convert("RGB") # ./heritageweaver/data/No_Image_Available.jpg
        #return Image.open(requests.get(img_uri, stream=True).raw).convert("RGB")
        
        return img
    
    return Image.open(img_uri).convert("RGB")
    
    
    

def plot_by_uri(img_uri: str) -> None:
    fig = plt.figure(figsize=(10, 10))
    img = open(img_uri)
    ax = fig.add_subplot(1, 1, 1,)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img)
    plt.show()

def plot_by_record(record: List[str]) -> None:
    
    fig = plt.figure(figsize=(10, 10))
    img = open_image(record[3]) 
    ax = fig.add_subplot(1, 1, 1,)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img)
    plt.show()
    print(soft_wrap_text(record[-1]))

   
def plotting_pairs(img_pair: Tuple[str, str]) -> None:
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 1
    
    for i in range(1, columns*rows +1):
        img = open_image(img_pair[i-1])

        ax = fig.add_subplot(rows, columns, i,)
        #title = f"{query_df.loc[i-1,'record_id']}" # 
        #ax.title.set_text(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
        
        
    plt.show()

