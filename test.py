import os
import glob
import torch
import librosa
import importlib
import yaml

import soundfile as sf
import pandas as pd
import numpy as np

from SpeakerNet import SpeakerNet
from backend import cosine_similarity_full


def load_audio(wavfile, device):
    audio, sr = librosa.load(wavfile, sr=16000)
    audio = torch.FloatTensor(audio).to(device)
    
    return audio

def load_model(config, ckpt_name, device):
    print('Setting Model...')
    
    feature_extractor = importlib.import_module('preprocessing.mel_transform').__getattribute__("feature_extractor")
    feature_extractor = feature_extractor(*config['FEATURE_EXTRACTOR'].values()).to(device)
    
    spec_aug = importlib.import_module('preprocessing.spec_aug').__getattribute__("spec_aug")
    spec_aug = spec_aug(*config['SPEC_AUG'].values()).to(device)
    
    model_cfg = config['MODEL']
    model = importlib.import_module('models.NeXt_TDNN').__getattribute__("MainModel")
    model =  model(
        depths = model_cfg['depths'], 
        dims = model_cfg['dims'],
        kernel_size = model_cfg['kernel_size'],
        block = model_cfg['block']).to(device)
    
    aggregation = importlib.import_module('aggregation.vap_bn_tanh_fc_bn').__getattribute__("Aggregation")
    aggregation = aggregation(*config['AGGREGATION'].values()).to(device)
    
    loss_function = importlib.import_module("loss.aamsoftmax").__getattribute__("LossFunction")
    loss_function = loss_function(*config['LOSS'].values())
    
    speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                             spec_aug = spec_aug, 
                             model = model,
                             aggregation = aggregation,
                             loss_function = loss_function).to(device)
    
    print('Load Pretrain Model..')
    checkpoint = torch.load(os.path.join(config['CHECKPOINT']['ckpt_path'], ckpt_name))
    speaker_net.load_state_dict(checkpoint["model"], strict=False)
    
    return speaker_net


if __name__=='__main__':
    THRESHOLD = 0.5
    
    print('Load yaml..')
    with open('K_NeXt_TDNN.yaml') as file:
        config = yaml.safe_load(file)
    ckpt_name = '00000.pt'
    
    enroll_files = glob.glob('enrollment/*.wav') ##speaker_.wav
    #test_audio = glob.glob('test/*.wav') #speaker_.wav
    test_audio_file = 'KHOtest.wav'
    
    model = load_model(config, ckpt_name, 'cuda')
    model.eval()
    
    test_audio = load_audio(test_audio_file, 'cuda')
    test_emb = model(test_audio.unsqueeze(0))
    
    score = {}
    with torch.no_grad():
        for enroll in enroll_files:
            enr_audio = load_audio(enroll, 'cuda')
            enr_emb = model(enr_audio.unsqueeze(0))
            similarity = cosine_similarity_full(enr_emb, test_emb)
            score[enroll[:-4]] = similarity #slicing .wav (to get the speaker)
        
    
    verification = max(score, key=score.get)
    
    if verification > THRESHOLD:
        print('OK')
    else:
        print('NO')