import torch
import os
import pandas as pd
import soundfile as sf
import tqdm
import librosa

from eval_metric import compute_eer, compute_MinDCF
from backend.cosine_similarity_full import cosine_similarity_full
from backend.euclidean_distance_full import euclidean_distance_full

# ===================
# Make Enrollment Data
# ===================
def make_enrollment(enr_df_path, val_df):
    """
    Make enrollment data
    
    Args:
        enr_df_path : enrollment df file (.pkl) path
        val_df : validation df file (.pkl) path
        
    Returns:
        enr_df : enrollment dataframe
    """
    enr_list = []
    enr_path = os.path.join(enr_df_path, 'enr_df.pkl')
    
    label_list = val_df.labels.unique()
    
    enr_df = pd.DataFrame()
    for label in tqdm.tqdm(label_list):
        cohorts = []
        for i in range(100):
            wavquery = val_df.query('labels == {0}'.format(label)).iloc[0]
            if i%2 == 0:
                cohort = val_df.query('labels == {0}'.format(label)).sample().wavfiles.values[0]
                cohort_label = 1
            else:
                cohort = val_df.query('labels != {0}'.format(label)).sample().wavfiles.values[0]
                cohort_label = 0
            wavquery['cohort'] = cohort
            wavquery['cohort_label'] = cohort_label
            cohorts.append(wavquery)
        cohort_df = pd.DataFrame(cohorts)
        enr_df = pd.concat([enr_df, cohort_df], ignore_index=True)
    
    enr_df.to_pickle(enr_path)
    
    del val_df
    del enr_list
    del label_list
    del cohort_df
    
    return enr_df

# ===================
# Validation
# ===================
def validation(model, base_path, device):
    """
    Model validation
    
    Args:
        model : verification model (speaker_net)
        base_path : df file path (train_df, enr_df)
        device : device (CPU/GPU)
        
    Returns:
        cos_eer : cosine eer
        euc_eer : euclidean eer
        cos_dcf : cosine MinDCF
        euc_dcf : euclidean MinDCF
    """
    model.eval()
    
    cos_sim_list = []
    euc_dist_list = []
    valid_label = []
    
    # ===================
    # Make/Load Enrollments
    # ===================
    if os.path.isfile(os.path.join(base_path, 'enr_df.pkl')):
        # if enrollment data exist
        enr_df = pd.read_pickle(os.path.join(base_path, 'enr_df.pkl'))
    else:
        # if enrollment data not exist
        val_df = pd.read_pickle(os.path.join(base_path, 'test_df.pkl'))
        enr_df = make_enrollment(base_path, val_df)
        
        del val_df
    
    # ===================
    # Model Validation
    # ===================
    print('Model Validation..')
    with torch.no_grad():
        for _, row in tqdm.tqdm(enr_df.iterrows(), total = enr_df.shape[0]):
            enr_x, _ = librosa.load(row['wavfiles'], sr=16000) #sf.read(row['wavfiles'])
            enr_x = torch.FloatTensor(enr_x)
            enr_emb = model(enr_x.unsqueeze(0).to(device))
            
            spk_x, _ = librosa.load(row['cohort'], sr=16000)# sf.read(row['cohort'])
            spk_x = torch.FloatTensor(spk_x)
            spk_emb = model(spk_x.unsqueeze(0).to(device))
            
            valid_label.append(row['cohort_label'])
            
            # cosine similarity
            cos_sim = cosine_similarity_full(enr_emb, spk_emb)
            cos_sim_list.append(cos_sim.detach().cpu().numpy())
            
            # Euclidean
            euc_dist = euclidean_distance_full(enr_emb, spk_emb)
            euc_dist_list.append(euc_dist.detach().cpu().numpy())
        
    # EER, DCF
    cos_eer, _ = compute_eer(cos_sim_list, valid_label)
    euc_eer, _ = compute_eer(euc_dist_list, valid_label)
    cos_dcf, _ = compute_MinDCF(cos_sim_list, valid_label)
    euc_dcf, _ = compute_MinDCF(cos_sim_list, valid_label)
    
    del enr_df
    del cos_sim_list
    del euc_dist_list
    del valid_label
    
    return cos_eer, euc_eer, cos_dcf, euc_dcf
