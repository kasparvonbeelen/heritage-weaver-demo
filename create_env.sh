cd /content/heritageweaver/
pip install -qr requirements.txt
# download data database and some additional metadata
cd /content/
pip install --upgrade --no-cache-dir gdown
gdown 19a91qQXxeU0zZqv49UrP7h2BZ82lEiUD
# unzip archive with vector database
unzip -qq ce_comms_db_copy.zip