sudo apt-get install libbz2-dev # Install bzip headers
sudo apt-get install libssl1.0-dev # Install openssl headers
conda install openssl==1.0.2n # Install local openssl to match
pip install --no-cache-dir zeroc-ice==3.6.5
python connect_to_omero.py

