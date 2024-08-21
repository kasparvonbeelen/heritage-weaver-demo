cd /content/heritageweaver/
pip install -qr requirements.txt
# download data database and some additional metadata
cd /content/
pip install --upgrade --no-cache-dir gdown
gdown 1r8ddjSuhtC4ZEmcEu15o5TV2dX35Qj12
# unzip archive with vector database
unzip -qq -o hw.zip