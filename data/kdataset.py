import os
import glob
import json
import itertools

import random
import torch
import numpy as np
import pandas as pd
import tqdm

import librosa
import soundfile as sf

from scipy import signal


# =====================================
# ASV Dataset
# =====================================
class asv_dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 asv_path,
                 df_path, 
                 rir_path, 
                 musan_path, 
                 max_frame,
                 augment,
                 train):
        
        # args
        self.asv_path = asv_path
        self.df_path = df_path
        self.rir_path = rir_path
        self.musan_path = musan_path
        self.augment = augment
        self.train = train
        
        # for train/test dataset
        if self.train:
            pkl_path = os.path.join(self.df_path, 'train_df.pkl')
        else:
            pkl_path = os.path.join(self.df_path, 'test_df.pkl')
        
        self.df = self.make_labels(pkl_path)
        self.wav_file_list = list(self.df['wavfiles'])
        self.label_list = list(self.df['labels'])
        
        self.max_frame = max_frame #300 -> 3 second audio
        self.max_length = max_frame*160+240
        
        # for reverberation
        if self.rir_path is not None:
            self.rir_files  = glob.glob(os.path.join(self.rir_path,'*/*/*.wav'))
        
        # for noise augmentation
        if self.musan_path is not None:
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
            self.noiselist = {
                'noise':[],
                'speech':[],
                'music':[]
            }
            
            musan_files = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            for file in musan_files:
                noise_type = file.split('\\')[-3] 
                self.noiselist[noise_type].append(file)
        
        # =====================================
        # dataset info
        # =====================================
        print('====== Dataset Load Info ======')
        print(f"Number of speakers : {len(self.df['labels'].unique())}") 
        print(f"Number of utterances : {len(self.df)}")
        print()
        print(self.df.info())
        print('===============================')
        
    def __len__(self):
        return len(self.df)
    
    def make_labels(self, pkl_path):
        """
        Make dataset labels
        
        Args:
            pkl_path : DataFrame (.pkl) file path
            
        Returns:
            df : DataFrame of loaded dataset
            - wavefiles : wave file path
            - labels : speaker label
        """
        if os.path.exists(pkl_path):
            # =====================================
            # Already exisit - just load dataframe
            # =====================================
            print('Read pkl...')
            print()
            df = pd.read_pickle(pkl_path)
            
        else:
            # =====================================
            # Does not exist - make dataframe
            # =====================================
            
            # make label/wav file list
            print('Get file list...')
            dataset_label_list = []
            dataset_label_list = glob.glob(os.path.join(self.asv_path, 'label/*/*/*/*.json'))
            
            dataset_wav_list = []
            dataset_wav_list = glob.glob(os.path.join(self.asv_path, 'wav/*/*/*/*.wav'))
            
            # save label/wav file list
            print('Save file list...')
            if self.train:
                with open(os.path.join(self.df_path, 'train_label_list.txt'), 'w+') as file:
                    file.write('\n'.join(dataset_label_list))
                
                with open(os.path.join(self.df_path, 'train_wav_list.txt'), 'w+') as file:
                    file.write('\n'.join(dataset_wav_list))
            else:
                with open(os.path.join(self.df_path, 'test_label_list.txt'), 'w+') as file:
                    file.write('\n'.join(dataset_label_list))
                
                with open(os.path.join(self.df_path, 'test_wav_list.txt'), 'w+') as file:
                    file.write('\n'.join(dataset_wav_list))
            
            # get speaker labels
            speaker_list = []
            for file in tqdm.tqdm(dataset_label_list):
                with open(file) as json_file:
                    json_data = json.load(json_file)
                tmp = pd.json_normalize(json_data)
                speaker_list.append(tmp['Speaker.SpeakerName'][0])
            
            speaker_unique_list = set(speaker_list)
            spk2idx = { spk : idx for idx, spk in enumerate(sorted(speaker_unique_list)) } # key: speaker / value: idx
            train_labels = [spk2idx[spk] for spk in speaker_list]
            
            # to DataFrame
            print('Save pkl...')
            print()
            df = pd.DataFrame()
            df['wavfiles'] = dataset_wav_list
            df['labels'] = train_labels
            df['speakers'] = speaker_list
            
            df.to_pickle(pkl_path) # save
            
            del dataset_wav_list
            del dataset_label_list
            del speaker_list            
        
        return df
    
    def add_revb(self, audio):
        """
        Reverb convolution using RIR dataset
        
        Args:
            audio : audio array (np.array)
            
        Returns:
            audio : reverberated audio (np.array)
        """
        # rir file random select
        rir_file = random.choice(self.rir_files)
        
        # read rir file
        rir, sr = sf.read(rir_file)
        rir = rir / np.sqrt(np.sum(rir**2))
        
        # reverberation
        waveform = signal.convolve(audio, rir, mode='full')
        
        return waveform[:self.max_length]
    
    def add_noise(self, noisecat, audio):
        """
        Noise augmentation using MUSAN dataset
        
        Args:
            noisecat : noise type (noise, speech, music)
            audio : audio array (np.array)
        
        Returns:
            audio : noise augmented audio array (np.array)
        """
        # compute db (decibel)
        val = max(0.0, np.mean(np.power(audio, 2)))
        clean_db = 10*np.log10(val+1e-4)
        
        # select noise
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        audio_length = len(audio)
        
        noises = []
        for noise in noiselist:
            # read noise file
            noiseaudio, sr  = sf.read(noise) # (noise_length, )
            noiseaudio = noiseaudio.astype(float)
            noise_length = len(noiseaudio)
            
            # zero-padding 
            if noise_length <= audio_length:
                shortage = audio_length - noise_length
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
                noiseaudio = noiseaudio.astype(float)
            else:
                start_frame = int(random.random()*(noise_length - audio_length))
                noiseaudio = noiseaudio[start_frame:start_frame + audio_length]
                
            # set snr
            noise_snr = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            val = max(0.0, np.mean(np.power(noiseaudio, 2)))
            noise_db = 10*np.log10(val+1e-4)
            
            noiseaudio = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            noises.append(noiseaudio)
            
        noise_stack = np.stack(noises,axis=0) # (numnoise, audio_length)
        noise_sum = np.sum(noise_stack,axis=0) # (audio_length,)
        
        return noise_sum + audio
    
    def __getitem__(self, idx):
        """
        Load wav files, padding, augmentation
        
        Returns:
            audio : audio tensor (torch.FloatTensor)
            label : speaker label
        """
        
        # load audio file
        wavfile = self.wav_file_list[idx]
        #audio, sr = sf.read(wavfile)
        audio, sr = librosa.load(wavfile, sr=16000)
        audio_len = audio.shape[0] # 41440
        
        # =====================================
        # only for training dataset
        # =====================================
        if self.train:
            # padding
            if audio_len <= self.max_length:
                pad_size = self.max_length-audio_len
                audio = np.pad(audio, (0,pad_size), 'wrap')
                audio = audio.astype(float)
            else:
                start = int(random.random()*(audio_len - self.max_length))
                audio = audio[start:start+self.max_length].astype(float) 
            
            # augmentation
            if self.augment:
                augtype = random.randint(0,5)
                if augtype == 1: # Reverberation
                    audio   = self.add_revb(audio)
                elif augtype == 2: # Music
                    audio   = self.add_noise('music',audio)
                elif augtype == 3: # Babble
                    audio   = self.add_noise('speech',audio)
                elif augtype == 4: # Noise
                    audio   = self.add_noise('noise',audio)
                elif augtype == 5: # TV noise
                    audio = self.add_noise('speech', audio)
                    audio = self.add_noise('music', audio)
        
        return torch.FloatTensor(audio), self.label_list[idx]
    
