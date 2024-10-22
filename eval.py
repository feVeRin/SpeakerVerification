import os
import torch
import librosa
import importlib

from SpeakerNet import SpeakerNet

# =====================================
# Audio Loader
# =====================================
def load_audio(wavfile, device):
    """
    Load wav files
    
    Args:
        wavfile : audio (.wav) file path
        device : device (CPU/GPU)
    
    Returns:
        audio : audio tensor
    """
    audio, sr = librosa.load(wavfile, sr=16000)
    audio = torch.FloatTensor(audio).to(device)
    
    return audio

# =====================================
# Model Loader
# =====================================
def load_model(config, ckpt_name, device):
    """
    Load Pretrained Verification Model
    
    Args:
        config : configuration (.yaml)
        ckpt_name : checkpoint file name
        device : device (CPU/GPU)
    
    Returns:
        speaker_net : pretrained verification model
    """
    print('Setting Model...')
    
    # feature extractor
    feature_extractor = importlib.import_module('preprocessing.mel_transform').__getattribute__("feature_extractor")
    feature_extractor = feature_extractor(*config['FEATURE_EXTRACTOR'].values()).to(device)
    
    # spectral augmentation
    spec_aug = importlib.import_module('preprocessing.spec_aug').__getattribute__("spec_aug")
    spec_aug = spec_aug(*config['SPEC_AUG'].values()).to(device)
    
    # TDNN model
    model_cfg = config['MODEL']
    model = importlib.import_module('models.NeXt_TDNN').__getattribute__("MainModel")
    model =  model(
        depths = model_cfg['depths'], 
        dims = model_cfg['dims'],
        kernel_size = model_cfg['kernel_size'],
        block = model_cfg['block']).to(device)
    
    # aggregation
    aggregation = importlib.import_module('aggregation.vap_bn_tanh_fc_bn').__getattribute__("Aggregation")
    aggregation = aggregation(*config['AGGREGATION'].values()).to(device)
    
    # loss function
    loss_function = importlib.import_module("loss.aamsoftmax").__getattribute__("LossFunction")
    loss_function = loss_function(*config['LOSS'].values())
    
    # speaker net wrapper
    speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                             spec_aug = spec_aug, 
                             model = model,
                             aggregation = aggregation,
                             loss_function = loss_function).to(device)
    
    # load .pt file
    print('Load Pretrain Model..')
    checkpoint = torch.load(os.path.join(config['CHECKPOINT']['ckpt_path'], ckpt_name))
    speaker_net.load_state_dict(checkpoint["model"], strict=False)
    #checkpoint = torch.load(r'C:\Users\jwjln\Desktop\SV\SpeakerVerification\experiments\NeXt_TDNN_light_C192_B1_K65\NeXt_TDNN_light_C192_B1_K65.pt')
    #speaker_net.load_state_dict(checkpoint["state_dict"], strict=False)
    
    return speaker_net
