import torch
import os
import pandas as pd
import soundfile as sf
import tqdm

from eval_metric import compute_eer
from backend.cosine_similarity_full import cosine_similarity_full
from backend.euclidean_distance_full import euclidean_distance_full


def make_enrollment(gt_model, enr_df_path, base_df):
    enr_list = []
    enr_path = os.path.join(enr_df_path, 'enr_df.pkl')
    
    label_list = base_df.labels.unique()
    
    enr_df = pd.DataFrame()
    for label in tqdm.tqdm(label_list):
        cohorts = []
        for i in range(100):
            wavquery = base_df.query('labels == {0}'.format(label)).iloc[0]
            if i%2 == 0:
                cohort = base_df.query('labels == {0}'.format(label)).sample().wavfiles.values[0]
                cohort_label = 1
            else:
                cohort = base_df.query('labels != {0}'.format(label)).sample().wavfiles.values[0]
                cohort_label = 0
            wavquery['cohort'] = cohort
            wavquery['cohort_label'] = cohort_label
            cohorts.append(wavquery)
        cohort_df = pd.DataFrame(cohorts)
        enr_df = pd.concat([enr_df, cohort_df], ignore_index=True)
    
    enr_df.to_pickle(enr_path)
    
    del base_df
    del enr_list
    del label_list
    del gt_model
    del cohort_df
    
    return enr_df

def validation(model, base_path, device):
    model.eval()
    
    cos_sim_list = []
    euc_dist_list = []
    valid_label = []
    
    if os.path.isfile(os.path.join(base_path, 'enr_df.pkl')):
        enr_df = pd.read_pickle(os.path.join(base_path, 'enr_df.pkl'))
    else:
        base_df = pd.read_pickle(os.path.join(base_path, 'train_df.pkl'))
        enr_df = make_enrollment(model, base_path, base_df)
        
        del base_df
    
    with torch.no_grad():
        for _, row in enr_df.iterrows():
            enr_x, _ = sf.read(row['wavfiles'])
            enr_x = torch.FloatTensor(enr_x)
            enr_emb = model(enr_x.to(device))
            
            spk_x, _ = sf.read(row['cohort'])
            spk_x = torch.FloatTensor(enr_x)
            spk_emb = model(spk_x.to(device))
            
            valid_label.append(row['cohort_label'])
            
            # cosine similarity
            cos_sim = cosine_similarity_full(enr_emb, spk_emb)
            cos_sim_list.append(cos_sim.detach().cpu().numpy())
            
            # Euclidean
            cos_sim = euclidean_distance_full(enr_emb, spk_emb)
            euc_dist_list.append(cos_sim.detach().cpu().numpy())
        
    # EER
    cos_eer, _ = compute_eer(cos_sim_list, valid_label)
    euc_eer, _ = compute_eer(euc_dist_list, valid_label)
    
    del enr_df
    del cos_sim_list
    del euc_dist_list
    del valid_label
    
    return cos_eer, euc_eer
