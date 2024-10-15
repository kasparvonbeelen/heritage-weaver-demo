cd /content/heritageweaver/
pip install -qr requirements.txt
# download data database and some additional metadata
cd /content/
pip install --upgrade --no-cache-dir gdown
gdown 1_JnLjNDZV-yv5BbRIxK2MeVxyvE-5o0U
# unzip archive with vector database
unzip -qq -o hw-08-10.zip