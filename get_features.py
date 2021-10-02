import requests


useful_features = ['acousticness', 'danceability', 'energy', 
                   'loudness', 'speechiness', 'valence', 
                   'key', 'mode']

def get_audio(ids, token):
    url = 'https://api.spotify.com/v1/audio-features'
    headers = {'Authorization': f'Bearer {token}'}
    params = {'ids': ','.join(ids.split())}
    r = requests.get(url, headers=headers, params=params)
    return r.json()['audio_features']

def get_features(features, attr):
    values = [v[attr] for v in features]
    print('\n'.join(map(str, values)))

def gen(features, attr):
    for att in attr:
        get_features(features, att)
        yield 


if __name__ == '__main__':
    ids = """
    16wAOAZ2OkqoIDN7TpChjR
7KXjTSCq5nL1LoYtL7XAwS
09nSCeCs6eYfAIJVfye1CE
4WQJPsbGOdqe24mUVHa8xb
0gMW8XpPFPjoApDii5Tj1u
4k9EAtZdZrzPlBUsFncXCZ
6v08G3CGcoyiODIWZoOxR4
67xBtV07CC73eFw7z5oCvU
7HMmFQsKsljwTw8bS7lu19
6fTt0CH2t0mdeB2N9XFG5r
5ghIJDpPoe3CfHMGu71E6T
4P5KoWXOxwuobLmHXLMobV
6Bo7elbKiRfaHBMJGEBHqF
0UiSGMxQHbCIEJjThfLuMn
13nvCPvVM3Y8dd5fu639B1
6lDx96uizO7bKiytk0t6KU
0r1x4VgTo0pGbXSw1hwmet
4P3LBZjsWULnAS08KebYfW
08bNPGLD8AhKpnnERrAc6G
6HJasLoTKvxglAMQH8nPcD
7FemQvLSKHlBYsIM3PoCs8
3fHxCZdMhWqYynQVMF4O3R
1jYiIOC5d6soxkJP81fxq2
03ZkvqZOANKndGXfAAPywG
7CkDERN0Gf63iamS4kP2sR
4EbAtdgNzR8So5cpNdyzut
5C2oCLLrtnYI8nNVOOruar
1iRvhKiXRElIH2Uf4gd95P
79r5vi5H3sYvnRNpkNylXP
5PDoWY5Av2Ba3rWIdY9Ij0
0PBSNKfGVRa6mexbVB6m47
1QulgzFcNHJswXIGev88wJ
1jwCtXGGek7YNYttuDpEMJ
5Zz8mrmVVhOq3iuv8Gh0MX
3jZDWsoxRHzzp3DYV8qeXE
7jR5gaq7wgmZU4DguTeeOc
4Y6VEDkRSpbn8Wt8x18RHh
0gOz9JUXsaKVzLTSmFDtdo
4P1L6AniTDJzANaubzWGYs
5aaxgcbaiwKTtvdGQ3FhbQ
7LZgdL0MxiElfaKZbuuE4l
1MONUudxAjEk76FJvzGhuD
0V5cvmTKsYmF5FmGGEAfmS
5c9qm0bMYawSyRNUTmUMs5
6iq7Jv43ULC1zRqtdpYDnI
7qH9Z4dJEN0l9bidizW7fq
4g0Gfk8H7mBcNAtvKMCnza
1tL1wrbBwzXpUhjItFytoY
7suXfwkW9Cg9fBS3San5T5
0xCA70t1ZA4fa9UOE0lIJm"""

    token = "BQBEQ89Kdyqx51D4n4sAP5shdBBAjN019_TFsHzdNcJIMHPn8_Z7oi0tpaAMRvJMMvlSKoNair07uA4dyi8jwfob40YuzTOMKsOj3Szb7vuiohyu1MC1LgAeoGkUGQVPPS-IiONI1_Wnp2zTXkrfxQjIzg"
    f = get_audio(ids, token)
    g = gen(f, useful_features)
    print("Type `next(g)` to continue.")