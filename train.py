import os
import importlib
import gc

import torch
import yaml
import librosa
import numpy as np
import matplotlib.pyplot as plt

from validation import *
import data.kdataset as kdataset
from SpeakerNet import SpeakerNet
from util import *

# ===================
# Training Wrapper
# ===================
def train(
    config, 
    max_epoch, 
    batch_size, 
    num_worker, 
    base_lr, 
    base_path, 
    device,
    ckpt = False,
    ckpt_name = None
    ):
    
    # dataset setting
    print('Setting Train Dataset...')
    asv_dataset = kdataset.asv_dataset(*config['TRAIN_DATASET'].values())
    
    train_loader = torch.utils.data.DataLoader(
        asv_dataset,
        batch_size = batch_size,
        num_workers = num_worker,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    # model setting
    print()
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
    
    optimizer = importlib.import_module("optimizer." + 'adamw').__getattribute__("Optimizer")
    optimizer = optimizer(speaker_net.parameters(), lr = base_lr*batch_size, weight_decay = 0.01,)
    
    scheduler = importlib.import_module("scheduler." + 'steplr').__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, step_size = 10, gamma = 0.8)
    
    # model summary
    print()
    print('===============================')
    print('Model Summary...')
    get_model_param_mmac(speaker_net, int(160*300 + 240), device)
    
    # model training
    print()
    print('===============================')
    print('Model Training...')
    
    if ckpt:
        # if checkpoint available
        print('Load Previous Checkpoint..')
        checkpoint = torch.load(os.path.join(config['CHECKPOINT']['ckpt_path'], ckpt_name))
        
        speaker_net.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ckpt_epoch = checkpoint["epoch"]
        
        # starts with chekpoint
        for epoch in range(ckpt_epoch+1, max_epoch):
            train_step(config, 
                       epoch, 
                       train_loader, 
                       speaker_net, 
                       optimizer, 
                       loss_function, 
                       scheduler, 
                       base_path, 
                       device)
    else:
        # scratch 
        for epoch in range(max_epoch):
            train_step(config, 
                       epoch, 
                       train_loader, 
                       speaker_net, 
                       optimizer, 
                       loss_function, 
                       scheduler, 
                       base_path, 
                       device)
    

# ===================
# Train Each Step
# ===================
def train_step(
    config, 
    epoch, 
    loader, 
    model, 
    optimizer, 
    loss_function, 
    scheduler, 
    base_path, 
    device):
    
    losses = 0
    model.train()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print('=== Epoch : {0} ==='.format(epoch))
    for idx, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        
        spk_emb = model(x.to(device))
        loss, _ = loss_function(spk_emb, y.to(device))
        losses += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if idx % 100 ==0:
            print('{0} step loss : {1}'.format(idx, loss))
    
    scheduler.step()
    print('-- Epoch {0} loss : {1}'.format(epoch, losses/len(loader)))
    
    # validation
    cos_eer, euc_eer = validation(model, base_path, device)
    print('Cosine EER : {0}, Euclidean EER : {1}'.format(cos_eer, euc_eer))
    
    ckpt_name = config['CHECKPOINT']['filename'].format(epoch)
    torch.save({'epoch' : epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'loss' : losses/len(loader),
                'cos_eer' : cos_eer,
                'euc_eer' : euc_eer,
                }, os.path.join(config['CHECKPOINT']['ckpt_path'], ckpt_name))
    print('-- Epoch {0} ckpt saved..'.format(epoch))
    print()
