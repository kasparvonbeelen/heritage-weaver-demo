cd /content/heritageweaver/
pip install -qr requirements.txt
!pip install --upgrade --no-cache-dir gdown
# download data database and some additional metadata
cd /content/
gdown 1egflWfa4R_ZHCTAyKlbElGeF8U66Z4Rr
# unzip archive with vector database
unzip -qq ce_comms_db_copy.zip