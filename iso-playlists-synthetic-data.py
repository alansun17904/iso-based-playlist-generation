import requests
from requests.oauthlib import OAuth1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fin = open("token", "r");
OAUTH = fin.readline().split("=")[1]
SOAR_ID = fin.readline().split("=")[1]
fin.close()

headers = {'Authorization': f'Bearer {OAUTH}'}
url = f'https://api.spotify.com/v1/users/{SOAR_ID}/playlists'
playlists = requests.get(url, headers=headers).json()
playlist_ids = [v['id'] for v in playlists['items']]
playlist_descriptions = [v['name'] + ' - ' + v['description'] for v in playlists['items']]

print(playlist_ids)
