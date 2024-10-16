import yaml

from train import train

if __name__ == '__main__':
    # parameter setting
    print('Load yaml..')
    
    with open('K_NeXt_TDNN.yaml') as file:
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
    
    # training
    train(config, MAX_EPOCH, BATCH_SIZE, NUM_WORKER, BASE_LR, BASE_PATH, DEVICE)
    
    # test
    