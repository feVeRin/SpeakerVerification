import yaml
import re
import glob
import torch

from train import train
from eval import load_model, load_audio
import backend.cosine_similarity_full as csf

if __name__ == '__main__':
    # =====================================
    # Parameter Setting
    # =====================================
    CKPT = True
    ckpt_name = 'ckpt_5.pt' ####
    
    print('Load yaml..')
    with open('./configs/K_NeXt_TDNN.yaml') as file:
        config = yaml.safe_load(file)
    
    BATCH_SIZE = config['PARAMS']['BATCH_SIZE']
    BASE_LR = float(config['PARAMS']['BASE_LR'])
    NUM_WORKER = config['PARAMS']['NUM_WORKER']
    CHANNEL_SIZE = config['PARAMS']['CHANNEL_SIZE']
    EMBEDDING_SIZE = config['PARAMS']['EMBEDDING_SIZE']
    MAX_FRAME = config['PARAMS']['MAX_FRAME']
    SAMPLING_RATE = config['PARAMS']['SAMPLING_RATE']
    MAX_EPOCH = config['PARAMS']['MAX_EPOCH']
    DEVICE = config['PARAMS']['DEVICE']
    BASE_PATH = config['PARAMS']['BASE_PATH']
    
    # =====================================
    # Train
    # =====================================
    if CKPT:
        train(config, MAX_EPOCH, BATCH_SIZE, NUM_WORKER, BASE_LR, BASE_PATH, DEVICE, ckpt=CKPT, ckpt_name=ckpt_name)
    else:
        train(config, MAX_EPOCH, BATCH_SIZE, NUM_WORKER, BASE_LR, BASE_PATH, DEVICE)
    
    # =====================================
    # Test
    # =====================================
    regex = re.compile(r'(?<=\\)(.*?)(?=.wav)')
    enroll_files = glob.glob('enrollment/*.wav') 
    test_audio_file = 'KHOtest.wav' ####
    
    model = load_model(config, ckpt_name, DEVICE)
    model.eval()
    
    test_audio = load_audio(test_audio_file, DEVICE)
    test_emb = model(test_audio.unsqueeze(0))
    
    # verification
    score = {}
    with torch.no_grad():
        for enroll in enroll_files:
            enr_audio = load_audio(enroll, DEVICE)
            enr_emb = model(enr_audio.unsqueeze(0))
            similarity = csf.cosine_similarity_full(enr_emb, test_emb)
            score[regex.search(enroll).group(0)] = similarity
        
    verification = max(score, key=score.get) # speaker name with max similarity 
    
    if verification == test_audio_file[:3]:
        print('!!!!! Verification Success (Speaker : {0}) !!!!!'.format(verification))
    else:
        print('!!!!! Verification Fail !!!!!')
