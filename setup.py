from setuptools import setup

setup(
    name='heritageweaver',
    version='0.0.1',    
    description='Python tools for the Congruence Engine Heritage Weaver Project',
    url='https://www.sciencemuseumgroup.org.uk/projects/the-congruence-engine',
    author='Kaspar Beelen',
    author_email='kaspar.beelen@sas.ac.uk',
    license='MIT',
    packages=['heritageweaver'],
    install_requires=['chromadb==0.4.22',
                    'datasets==2.7.1',
                    'keybert==0.7.0',
                    'matplotlib==3.7.2',
                    'numpy==1.24.3',
                    'pandas==1.3.5',
                    'pillow==10.2.0',
                    'requests==2.31.0',
                    'seaborn==0.13.2',
                    'sentence_transformers==2.2.2',
                    'spacy',
                    'tensorboard==2.13.0',
                    'tensorflow==2.13.0',
                    'tensorflow_macos==2.13.0',
                    'torch==2.0.1',
                    'torchvision==0.15.2',
                    'tqdm==4.65.0',
                    'transformers==4.37.1'                   
                      ],

    classifiers=[
        'Development Status :: 0 - Experiment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',  
        'Operating System :: POSIX :: MacOS', 
        'Programming Language :: Python :: 3.9',
    ],
)