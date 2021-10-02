"""
Alan Sun 
08.07.21 
ISO-Based Deep Learning Using LSTMs Model API 

This script serves as an executive from which specific functionalities of the 
model. The main endpoint that is established is the `get_recommendations` 
function, which returns a list of songs, that is recommended by the AI engine
based on the emotional mood descriptors that are passed into the model. We note
that the script also contains endpoints for training and loading pre-trained 
images.  
"""


import json
import math
import torch
import random
import pickle
import requests
import datetime
import collections
import transformers
import threading
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from musixmatch import Musixmatch
from transformers import pipeline
from sklearn import decomposition
from scipy.special import softmax
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
    
    def __len__(self):
        return len(self.stoi)
    
    def fit_on_moods(self, moods):
        flat = []
        
        Tokenizer.flatten(moods, flat)
        vocab = sorted(set(flat))
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for index, word in enumerate(vocab):
            self.stoi[word] = index
        self.itos = {v : k for k, v in self.stoi.items()}

    def flatten(l, flat):
        """
        Recursively, flatten a list.
        """
        if type(l) != list:
            flat.append(l)
        else:
            for el in l:
                Tokenizer.flatten(el, flat)

    def moods_to_token(self, states, reverse=False):
        """
        Recursively tokenize moods, while preserving the
        structure of the list. When `reverse` is true, the
        method translates the tokens back into the mood strings
        """
        if type(states) != list:
            if reverse:
                return self.itos[states]
            else:
                return self.stoi[states]
        else:
            for index, state in enumerate(states):
                states[index] = self.moods_to_token(state, reverse)
            return states
tokenizer = torch.load('data/tokenizer.pth')

class Compose:
    """
    A helper class that connects all of the possible data augmentations, and
    executes them during training. 
    """
    def __init__(self, transformations):
        self.transform = transformations
    
    def __call__(self, moods, features):
        for trans in self.transform:
            moods, features = trans(moods, features)
        return moods, features


class FeatureProtuberance:
    def __init__(self, max_protuberance, phi):
        """
        :param max_protuberance: the maximum percentage of protuberance.
        If 0.5 is given then each component, c, in the feature matrix 
        will have a potential new min/max of c +- 0.5 * c.
        :param phi: the probability that a given component is going to
        be augmented. 
        """
        self.protuberance = max_protuberance
        self.phi = phi
    
    def __call__(self, moods, features):
        pct = (torch.randn(features.size()) - 0.5)
        pct = pct * self.phi
        aug = torch.randn(features.size()) > self.phi
        return moods, features + aug * pct * features


class Reverse:
    def __init__(self, phi):
        """
        :param phi: (0, 1), the probability that the mood states and 
        features will be reversed.
        """
        self.phi = phi
        
    def __call__(self, moods, features):
        if random.random() > self.phi:
            return moods, features
        return (torch.flip(moods, dims=(0,)),
                torch.flip(features, dims=(0,)))


class ISODataset(Dataset):
    """
    The `ISODataset` class packages training data into a single index-able object.
    This makes it easy for torch to use as a generator.
    """
    def __init__(self, directory, maxlen=5, transform=None, batch_size=0):
        """
        Initializer.
        :param maxlen: The reader should note that this is the maximum number
        of mood transitions there can be. The constants (5) proceeding this
        block represent the number of descriptors allowed for each mood state.
        """
        self.pca = None
        self.n_comp = 11
        self.components = np.array([])
        self.mean = np.array([])
        self.directory = directory
        self.df = None
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.transform = transform
        try:
            self.df = pd.read_csv(self.directory, index_col=0)
            self.df = self.df[self.df['features'] != '[null]']
        except FileNotFoundError:
            raise FileNotFoundError(f"The training data:  {self.directory}, was not found.")
    
    def pca_reduction(self, percent_var=0.95):
        """
        Reduce the dimensionality of the data from 11->n. Where
        the sum of the average percentage variances calculated by
        eig / tr(D), where D is the diagonal matrix of eigenvalues
        is greater than `percent_var`. It is important to note that 
        we assume a dataframe where the initial values are still in json
        form. 
        :param var: must be greater than 0 and less than 1. 
        The method then uses these eigenvectors to reduce the dimensions
        of the audio features, and returns the number of components being
        used as well as the matrix of eigenvectors. 
        """
        # We proceed by stacking all the songs into a large matrix
        all_playlists = [json.loads(self.df.iloc[entry]['features'])
                         for entry in range(len(self.df))]
        all_playlists = np.vstack(all_playlists)
        self.mean = torch.mean(torch.from_numpy(all_playlists), 0)
        self.pca = decomposition.PCA()
        self.pca.fit(all_playlists)
        
        p_singular = self.pca.singular_values_ / sum(self.pca.singular_values_)
        counter, i = 0, 0
        while i < percent_var:
            i += p_singular[counter]
            counter += 1
        self.n_comp = counter

        self.components = torch.from_numpy(self.pca.components_[:self.n_comp,:]).float()
        return self.pca.components_[:counter,:], counter
        
    def pca_reconstruction(self, y):
        """
        This method converts a given matrix of predicted `y` datapoints
        which are in reduced PCA form back into their `full` features 
        using PCA reconstruction. This is done by simply right-multiplying
        `y` by our eigenvectors, and the adding the mean.
        """
        Xhat = torch.matmul(y, self.components)
        return Xhat + self.mean.float()
        
    def __len__(self):
        return max(len(self.df), self.batch_size)
    
    def __getitem__(self, idx):
        idx = idx % len(self.df)
        mood_states = json.loads(self.df.iloc[idx]['moods_states'])
        audio_features = self.df.iloc[idx]['features']
        audio_features = torch.Tensor(json.loads(audio_features))
        for index, state in enumerate(mood_states):
            mood_states[index] = np.pad(state, (0,5-len(state)), 
                                        constant_values=tokenizer.stoi['<pad>'])
        while len(mood_states) < 5:
            mood_states.append(np.full(5, tokenizer.stoi['<pad>']))
        mood_states = torch.LongTensor(mood_states)
        
        # augmentations
        if self.transform:
            return self.transform(mood_states, audio_features)
        return mood_states, audio_features


class TestDataset(Dataset):
    """
    The `ISODataset` class packages training data into a single index-able object.
    This makes it easy for torch to use as a generator.
    """
    def __init__(self, df, batch_size=0):
        """
        Initializer.
        :param maxlen: The reader should note that this is the maximum number
        of mood transitions there can be. The constants (5) proceeding this
        block represent the number of descriptors allowed for each mood state.
        """
        self.df = df
        self.batch_size = batch_size
        
    def __len__(self):
        return max(len(self.df), self.batch_size)
    
    def __getitem__(self, idx):
        idx = idx % len(self.df)
        mood_states = json.loads(self.df.iloc[idx]['moods_states'])
        for index, state in enumerate(mood_states):
            mood_states[index] = np.pad(state, (0,5-len(state)), 
                                        constant_values=tokenizer.stoi['<pad>'])
        while len(mood_states) < 5:
            mood_states.append(np.full(5, tokenizer.stoi['<pad>']))
        mood_states = torch.LongTensor(mood_states)
        
        return mood_states, self.df.iloc[idx]['length']


def iso_collate(batch):
    moods, features, lengths = [], [], []
    for data_point in batch:
        moods.append(data_point[0])
        features.append(data_point[1])
        lengths.append(len(data_point[1]))
    features = pad_sequence(features, batch_first=True)
    moods = pad_sequence(moods, batch_first=True, padding_value=tokenizer.stoi['<pad>'])
    return moods, torch.LongTensor(lengths), features


def test_collate(batch):
    moods, lengths = [], []
    for data_point in batch:
        moods.append(data_point[0])
        lengths.append(data_point[1])
    moods = pad_sequence(moods, batch_first=True, padding_value=tokenizer.stoi['<pad>'])
    return moods, torch.LongTensor(lengths)

# Model Architecture

class Attention(pl.LightningModule):
    """
    The attention mechanism of the network. On each time step, of the LSTM,
    the LSTM cell looks at the previous hidden state as well as the input
    to the LSTM, then it weights the various dimensions of the input based
    on the hidden state / input. This is done by applying two linear on the
    hidden and input states respectively, then combining the outputs, running
    them through another linear layer, and interpolating the final weights 
    using the softmax function. The result of the attention layer, is a 
    sum product of all the weights and the respective attributes.
    """
    def __init__(self, embed_dim, attention_dim=40, maxlen=5, output_dim=11):
        """
        Initializer for the attention mechanism of the network.
        :embed_dim: the dimension of the embeddings – hyperparamters.
        :param attention_dim: specifies dimension of the hidden attention
        layer. This is simply a hyperparameter and will only affect the
        efficacy of the network, not its functionality. 
        :param maxlen: specifies the maximum number of mood transitions allowed.
        :param output_dim: the dimension of the output, this varies as we 
        include/exclude prediction feature. We note that to predict all of the features,
        we simply use the default value of 11.
        """
        super().__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.mood_attention1 = nn.Linear(self.embed_dim, self.attention_dim)
        self.mood_attention2 = nn.Linear(self.attention_dim, self.attention_dim)
        self.hidden_attention1 = nn.Linear(output_dim, self.attention_dim)
        self.hidden_attention2 = nn.Linear(self.attention_dim, self.attention_dim)
        # the input to the hidden attention is 11 as that is the size of the
        # desired output dimension.
        self.attention = nn.Linear(self.attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, moods, hidden):
        """
        :param moods: The raw input mood states – this is of size (bs x maxlen x 5 x embed_dim).
        The reader should note that this *needs* to be preprocessed into the size
        (bs x (maxlen * 5) x embed_dim). 
        :param hidden: The previous hidden state of the LSTM cell – this should be of
        size (bs x 10).
        The result of this function is to find a weighting, or alternatively, where to 
        pay "attention" to based on the `moods` and `hidden` state. The weights
        of the attention, `alpha` of size (bs x (maxlen * 5)), is the used to in a sum product
        with the moods (bs x (maxlen * 5) x embed_dim), yielding a size of (bs x embed_dim).
        This single vector then acts as the inputs to the LSTM cell. 
        """
        att1 = self.relu(self.mood_attention1(moods))
        att1 = self.mood_attention2(att1)
        att2 = self.relu(self.hidden_attention1(hidden))
        att2 = self.hidden_attention2(att2)
        att = self.attention(self.relu(att1 + att2.unsqueeze(1)))
        alpha = self.softmax(att)
        weighted_moods = (moods * alpha).sum(dim=1)
        return weighted_moods, alpha


class Model(pl.LightningModule):
    """
    An attention-based, one-directional baseline model.
    """
    def __init__(self, tokenizer=tokenizer, dropout=0.0, maxlen=5, embed_dim=3, lr=1e-2,
                 weight_decay=1e-9, hidden_dim=11, output_dim=11,
                 dataset=None):
        """
        Initializer.
        :param tokenizer: the tokenizer used to create the tokenizations for
        the mood states and descriptors. 
        :param dropout: the probability of dropout of the layer between the 
        hidden state and the final output of each LSTM cell.
        :param maxlen: the maximum number of mood transition states that are allowed. 
        It should be noted that this must be greater than any of the number of the 
        states associated with each datapoint; otherwise, it will cause errors. 
        :param embed_dim: the dimensionality of each embedding.
        :param lr: learning rate of the network, TODO: implement separate learning rates
        for the attention network and the LSTM cell.
        :param output_dim: the dimension of the output, this varies as we 
        include/exclude prediction feature. We note that to predict all of the features,
        we simply use the default value of 11.
        """
        super().__init__()
        self.lr = lr
        self.bce = nn.BCELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.weight_decay = weight_decay
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.dataset = dataset
        self.embedding = nn.Embedding(len(self.tokenizer.itos), self.embed_dim, 
                                      padding_idx=self.tokenizer.stoi['<pad>'])
        self.attention = Attention(self.embed_dim, output_dim=hidden_dim)
        self.h0 = nn.Linear(maxlen * self.embed_dim * 5, 
                            self.hidden_dim)
        self.c0 = nn.Linear(maxlen * self.embed_dim * 5,
                            self.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.forget = nn.Linear(self.hidden_dim, self.embed_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
    def init_hidden_states(self, x):
        """
        Given the mood states: flattened into a (bs x (maxlen * 5 * 3)) vector,
        we use two separate linear layers to find the initial cell state and 
        initial hidden state. This is necessary over simply random initialization,
        as the attention associated with the first cell is dependnet on h0.
        """
        return self.h0(x), self.c0(x)
        
    def forward(self, x):
        """
        Forward feeding of the model. The network proceeds by first converting the mood
        states into their respective embedding representations. Then, the flattened inputs
        are used to determine the initial hidden/cell states of the given LSTM. The given
        inputs are then sorted based on the length of their outputs. This makes it easier
        for prediction. 

        :param x: a tuple that contains three items, the first being the various
        mood states that are being queried, the second being the lenghts of the
        desired labels of each datapoint, and finally the features – the target 
        outputs. 
        """
        mood_states, lengths, audio_features = x
        bs = mood_states.size(0)
        mood_states = self.embedding(mood_states)
        moods = mood_states.view(bs, (self.maxlen * 5), self.embed_dim)
        
        sorted_lengths, indicies = lengths.sort(dim=0, descending=True)
        moods, audio_features = moods[indicies], audio_features[indicies]
        h, c = self.init_hidden_states(moods.view(bs, -1))  # (bs x output_dim)

        predictions = torch.zeros(bs, max(lengths), self.output_dim)
        for timestep in range(max(lengths)):
            num_predict = sum([l > timestep for l in lengths])
            attention_weighted_moods, alphas = self.attention(moods[:num_predict], 
                                                      h[:num_predict])
            gate = self.sigmoid(self.forget(h[:num_predict]))
            weighted_moods = gate * attention_weighted_moods
            h, c = self.lstm(weighted_moods, 
                             (h[:num_predict], c[:num_predict]))
            preds = self.fc(self.dropout(h))
            predictions[:num_predict, timestep, :] = preds
        return self.sigmoid_relevant(predictions), audio_features
    
    def sigmoid_relevant(self, predictions):
        # entropy loss for attributes (acousticness, 0), (danceability, 1), (energy, 2), (instrumentalness, 3)
        # (liveness, 5), (loudness, 6), (speechiness, 8), (valence, 10)
        loss = 0
        for attr in [0, 1, 2, 3, 5, 6, 8, 10]:
            predictions[:,:,attr] = F.sigmoid(predictions[:,:,attr])
        return predictions
    
    def step(self, batch, batch_idx):
        """
        One "step" of the model. 
        """
        predictions, targets = self(batch)
        if self.dataset is not None:
            predictions = self.dataset.pca_reconstruction(predictions)
        loss = self.entropy_loss(predictions, targets)
        return loss, {'loss': loss}

    def entropy_loss(self, predictions, targets):
        # entropy loss for attributes (acousticness, 0), (danceability, 1), (energy, 2), (instrumentalness, 3)
        # (liveness, 5), (loudness, 6), (speechiness, 8), (valence, 10)
        loss = 0
        for attr in [0, 1, 2, 3, 5, 6, 8, 10]:
            attr_loss = self.bce(predictions[:,:,attr], targets[:,:,attr])
            loss += abs(attr_loss).sum(axis=1).mean()
        # mse loss for attributes (key 4), (mode 7), (tempo 9)
        for attr in [4, 7, 9]:
            attr_loss = self.mse(predictions[:,:,attr], targets[:,:,attr])
            loss += attr_loss.sum(axis=1).mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'train_{k}': v for k, v in logs.items()},
                      on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'val_{k}': v for k, v in logs.items()}, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        mood_states, lengths = batch
        bs = mood_states.size(0)
        mood_states = self.embedding(mood_states)
        moods = mood_states.view(bs, (self.maxlen * 5), self.embed_dim)
        
        sorted_lengths, indicies = lengths.sort(dim=0, descending=True)
        moods = moods[indicies]
        h, c = self.init_hidden_states(moods.view(bs, -1))  # (bs x output_dim)

        predictions = torch.zeros(bs, max(lengths), self.output_dim)
        for timestep in range(max(lengths)):
            num_predict = sum([l > timestep for l in lengths])
            attention_weighted_moods, alphas = self.attention(moods[:num_predict], 
                                                      h[:num_predict])
            gate = self.sigmoid(self.forget(h[:num_predict]))
            weighted_moods = gate * attention_weighted_moods
            h, c = self.lstm(weighted_moods, 
                             (h[:num_predict], c[:num_predict]))
            preds = self.fc(self.dropout(h))
            
            predictions[:num_predict, timestep, :] = preds
        return self.sigmoid_relevant(predictions)
    
    def configure_optimizers(self):
        """
        Configuration of the optimizer used to train the model.
        This method is implicitly called by torch lightning during training. 
        Note that the learning rate and weight decay is given by the initialization
        parameters of the model. 
        """
        return (optim.Adam(self.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay))


def train_model(stored_directory, training_directory, epochs, batch_size, steps):
    """
    The model is stored in a directory made for model weights, and it is named 
    based on the time and date that it finished training. 
    """
    transform = Compose([
    #     FeatureProtuberance(0.10, 0.5),
        Reverse(0.3)
    ])


    iso = ISODataset(training_directory, transform=transform,
                     batch_size=steps)
    train_loader = DataLoader(iso,
                            batch_size=batch_size,
                            collate_fn=iso_collate)
    model = Model(tokenizer, embed_dim=64, hidden_dim=256, 
                dropout=0.2, lr=1e-3, weight_decay=1e-6)
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader)
    n = datetime.datetime.today()
    torch.save(model.state_dict(), f'models/baseline-{n.month}-{n.day}.pth')


def predict(stored_directory, moods, lengths):
    """
    :param moods: Needs to be a three dimensional list. It can contain many
    independent mood transitions and the engine will be able to spit out
    predictions for all of these. Each mood transition needs to come
    in the form of a two dimesnional list:
    [
     ['sad', 'lonely'], 
     ['serene', 'relaxed'], 
     ['powerful', 'energetic']
    ]
    where each sublist represents a mood state and the words inside of this
    list are mood descriptors that describe the mood state. A mood state
    can have a maximum of five mood descriptors and a playlist can 
    have a maximum of five mood states. 
    :param lengths: A list of integers describing the required length of each
    respective playlist in the three-dimensional mood list. 
    """
    model = Model(tokenizer=tokenizer, embed_dim=64, hidden_dim=256, 
                dropout=0.2, lr=1e-3, weight_decay=1e-6)
    model.load_state_dict(torch.load(stored_directory))

    mood_states = [json.dumps(tokenizer.moods_to_token(v)) for v in moods]
    rows = [[v, lengths[i]] for i, v in enumerate(mood_states)]
    dtf = pd.DataFrame(rows, columns=['moods_states', 'length'])
    test = TestDataset(dtf.copy(), batch_size=1)
    test_loader = DataLoader(test,
                            batch_size=1,
                            collate_fn=test_collate)
    model.eval()
    results = []
    for batch_idx, batch in enumerate(test_loader):
        output = model.test_step(batch, batch_idx)
        output = output.reshape(output.shape[1], output.shape[2])
        results.append(output)

    # Convert each song and each respective playlist into a list.
    results = [v.detach().numpy().tolist() for v in results]
    return results

def get_recommendation(feature, artists, tracks, genres,
                       playlist, order, lock, beam_width, api_token='api'):
    API_TOKEN = open('api').readline()
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    url = "https://api.spotify.com/v1/recommendations"
    params = {}
    params['min_acousticness'] = feature[0]
    params['target_danceability'] = feature[1]
    params['target_energy'] = feature[2]
    # params['target_instrumentalness'] = feature[3]
    # params['target_liveness'] = feature[5]
    params['max_loudness'] = feature[6]
    params['target_speechiness'] = feature[8]
    params['target_valence'] = feature[10]
    # round integer fields -> 
    # params['target_key'] = round(feature[4])
    # params['target_mode'] = round(feature[7])
    # params['target_tempo'] = round(feature[9])
    # get seed artists, tracks, and genres
    params['seed_artists'] = artists
    params['seed_tracks'] = tracks
    params['seed_genres'] = genres
    params['limit'] = 10
    # get id of recommended track
    reqst = requests.get(url, headers=headers, params=params)
    reqst = reqst.json()
    tracks_gotten = len(reqst['tracks'])
    track_ids = [reqst['tracks'][i]['id'] 
                 for i in range(min(tracks_gotten, beam_width))]
    # store track id in appropriate index in dict
    lock.acquire()
    playlist[order] = ','.join(track_ids)
    lock.release()

def get_track_name(id):
    params = {'ids': id}
    API_TOKEN = open('api').readline()
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    req = requests.get('https://api.spotify.com/v1/tracks', 
                       headers=headers, params=params).json()
    tracks = []
    for i in range(len(req['tracks'])):
        tracks.append((req['tracks'][i]['name'],
                       req['tracks'][i]['artists'][0]['name']))
    return tracks

def get_lyrics(name, artist):
    musix = Musixmatch(open('musix_api').readline())
    response = musix.matcher_lyrics_get(name, artist)
    if response['message']['header']['status_code'] != 200:
        print(f'{name} by {artist} lyrics could not be found.')
        return None
    else:
        return response['message']['body']['lyrics']['lyrics_body']

def sentiment_analysis(lyrics, pipeline):
    results = pipeline(lyrics)
    results = results[0]
    score = results['label']
    prob = results['score']
    if score == 'NEGATIVE':
        prob = -prob

    # rescale [-1, 1] to [0, 1] using compressed sigmoid
    return 1 / (1 + math.exp(-math.e * 1.4 * prob))
    

def translate_features(features, api='api', artists='66CXWjxzNUsdJxJ2JdwvnR', 
                       seed_songs='463CkQjx2Zk1yXoBuierM9', 
                       seed_genre='pop,indie,folk'):
    # features contains a list of a playlists
    # this which then contains a list of songs
    playlist = {}
    # Create threads
    threads = []
    lock = threading.Lock()
    for index, song in enumerate(features):
        threads.append(threading.Thread(target=get_recommendation,
                                        args=(song, artists, seed_songs, 
                                            seed_genre, playlist,
                                            index, lock, 10)))
    # Start threads
    for thread in threads:
        thread.start()

    # Join threads
    for thread in threads:
        thread.join()
    od = collections.OrderedDict(sorted(playlist.items()))
    for k, v in od.items():
        print(f'{str(k+1).ljust(3)}: {v}')
    return od

def compose_playlist(od, valences, pipeline):
    playlist = []
    possibilities = []
    for key, value in od.items():
        names = get_track_name(value)
        scores = []
        for name in names:
            lyrics = get_lyrics(name[0], name[1])
            if lyrics is None:
                continue
            else:
                sentiment = sentiment_analysis(lyrics, pipeline)
                losses = loss(playlist, name[0], valences[key], sentiment)
                scores.append((name[0], name[1], losses))
        scores.sort(key=lambda x: x[2])
        playlist.append(scores[0])
        possibilities.append(scores)
    return playlist

bce = torch.nn.BCELoss()

def loss(existing_list, song, valence, sentiment):
    count = 1
    for exists in existing_list:
        if exists[0] == song:
            count += 1
    return count * bce(torch.Tensor([valence]), 
                       torch.Tensor([sentiment])) + 10 * (count - 1)


def get_scaled_valences(features):
    valences = [f[10] for f in features]
    mini = min(valences)
    maxi = max(valences)
    scaling = lambda x: (x - mini) / (maxi - mini)
    return [scaling(v) for v in valences]

def min_max_scale_features(features):
    ["acousticness", "danceability", "energy", "instrumentalness", 
         "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
    # scale (acousticness, 0), (danceability, 1), (energy, 2), (instrumentalness, 3)
    # (liveness, 5), (loudness, 6), (speechiness, 8), (valence, 10)
    feats = np.array(features)
    new_feats = []
    for feat in feats:
        for column in [0, 1, 2, 3, 5, 6, 8, 10]:
            c = feat[:, column]
            feat[:, column] = (c - min(c)) / (max(c) - min(c))
        new_feats.append(feat.tolist())
    return new_feats


if __name__ == '__main__':
    pipe = pipeline('sentiment-analysis')
    features = predict('models/modified-loss-8-11.pth', 
        [[['sad', 'lonely'], ['happy', 'joyful']]], 
        [20])
    print(features)
    # scaled_features = min_max_scale_features(features)
    od = translate_features(features[0])
    print(compose_playlist(od, get_scaled_valences(features[0]), pipe))