from PIL import Image
import matplotlib.pyplot as plt
import requests

def open_image(img_uri):
    if img_uri.startswith('http'):
        if 'sciencemuseum' in img_uri:
            img_uri = img_uri.replace('.uk/images/','.uk/').lower()
        return Image.open(requests.get(img_uri, stream=True).raw).convert("RGB")
    return Image.open(img_uri)
    
    

def plot_by_uri(img_uri):
    fig = plt.figure(figsize=(10, 10))
    img = open(img_uri)
    ax = fig.add_subplot(1, 1, 1,)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img)
    plt.show()

def plot_by_record(record):
    fig = plt.figure(figsize=(10, 10))
    img = open_image(record[-2]) 
    ax = fig.add_subplot(1, 1, 1,)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img)
    plt.show()
    print(record[-1])

   
def plotting_pairs(img_pair):
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

def soft_wrap_text(text, max_char_length=40):
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