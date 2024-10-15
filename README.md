# heritage-weaver
Code for Weaving Heritage Collections. Congruence Engine Technical Investigations

## Run on Colab


### Query the Collections
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kasparvonbeelen/heritageweaver/blob/2-vdb/colab_notebooks/Query.ipynb)

### Search and Annotate
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kasparvonbeelen/heritageweaver/blob/2-vdb/colab_notebooks/Annotate.ipynb)

### Annotate Links
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kasparvonbeelen/heritageweaver/blob/2-vdb/colab_notebooks/Annotate-Links.ipynb)

## Instructions for Local Installation 

```
conda create -n heritageweaver python=3.9
```

```
conda activate heritageweaver
```

```
cd path/to/heritageweaver
```

```
pip install -r requirements.txt
```

```
conda install -c anaconda ipykernel
```

```
python -m ipykernel install --user --name=heritageweaver
```