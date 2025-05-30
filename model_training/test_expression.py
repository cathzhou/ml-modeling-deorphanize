# %%
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import time
import urllib.parse
# %%
uniprot_id = 'P50052' #AGTR2

# Build the search URL with parameters
params = {
    'search': uniprot_id,
    'format': 'json',
    'columns': 'g,eg,up,rnascd,rnascsm',
    'compress': 'no'
}
base_url = "https://www.proteinatlas.org"

search_url = f"{base_url}/api/search_download.php?{urllib.parse.urlencode(params)}"
print(f"Searching for UniProt ID: {uniprot_id}")
print(f"URL: {search_url}")
response = requests.get(search_url)
response.raise_for_status()
data = response.json()
# %%
            if not data:
                print(f"No data found for UniProt ID: {uniprot_id}")
                return None
            
            # Find the matching entry with our UniProt ID
            matching_entries = [entry for entry in data if entry.get('up') == uniprot_id]
            if not matching_entries:
                print(f"No exact match found for UniProt ID: {uniprot_id}")
                return None
            