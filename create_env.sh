cd /content/heritageweaver/
pip install -qr requirements.txt
# download data database and some additional metadata
gdown 1ZKZhmjVDO2U2cYRmCvu_phGgQAFY7__R
gdown 1BkIfIxksVjXHeWYj2JM-dYR6Ibprz8SN
# unzip archive with vector database
unzip -qq ce_comms_db.zip
kill -9