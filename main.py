# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
#%% Initial Settings
# Comment/uncomment the following based on your debugging needs
DATA2USE="SwissMetro" #['SwissMetro', 'Synthesized']
STUDYNAME="dcmseal2" #e.g., dcmseal2, explore4; 'explore' will trigger grid search and the last number (if there is) will override # latent classes
NUMTRIALS=50 if "explore" in STUDYNAME else 200
LOGNUM=1 if "explore" in STUDYNAME else 10
# see results with anaconda: tensorboard --logdir 'LogSavedDirectory'/

import os
from pathlib import Path
import numpy as np
import math
import warnings
import shutil
from itertools import combinations
import re
import pandas as pd
import optuna
from functools import partial

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Grad strides do not match.*")

if '__file__' in globals():
    WPATH = Path(__file__).resolve().parent
else:
    WPATH=Path('D:/My Drive/phd/LabETC/DCMSEAL/GitHub').resolve() # input your own directory for smoother IDE experience
os.chdir(WPATH)

if globals().get('DATA2USE') is None:
    DATASETS=[p.name for p in (WPATH/'data').iterdir() if p.is_dir()]
    DATA2USE=input(f"Type a dataset you would like to use from {DATASETS} (case-sensitive): ")

match DATA2USE:
    case 'Synthesized':
        train_size= 25000
        CONFIG = {
            # -- Data Processing Hyperparameters --
            'train_size':25000,
            "core_vars": ["x1","x2","x3"],
            "non_positive_core_vars":["x1","x2","x3"],
            "embedding_vars": ["e1","e2"],
            "segmentation_vars_categorical": ["s1","s2","o1"],
            "test_size": 0.2,
            "random_state": 5723588,
            # -- Model Architecture Hyperparameters --
            "n_latent_classes": 3,
            "n_alternatives": 2,
            "choice_mode": "heterogeneous",
        }
    case 'SwissMetro':
        CONFIG = {
            # -- Data Processing Hyperparameters --
            'train_size':9036*3,
            "core_vars": ["CO", "TT", "HE"],
            "non_positive_core_vars":["CO","TT","HE"],
            "embedding_vars": ["firstclass","tickettype","whopaid","luggage","seats"],
            "segmentation_vars_categorical":["age","male","income","seasonalticket","purpose","oricanton","descanton"], 
            "segmentation_vars_continuous": [],
            "test_size": 0.2,
            "random_state": 5723588,
            # -- Model Architecture Hyperparameters --
            "n_latent_classes": 2,
            "n_alternatives": 3,
            "choice_mode": "heterogeneous",
        }
# For a grid search that outputs a list "pools" of tuples where each tuple is ([seg_list],[emb_list])
if STUDYNAME[:-1].lower()=='explore':
    pool = sorted(CONFIG['embedding_vars']+CONFIG['segmentation_vars_categorical'])
    pools = [(list(s), [x for x in pool if x not in s]) 
          for r in range(len(pool)+1) 
          for s in combinations(pool, r)]
    rows = [{el: int(el in first_list) for el in pool}
        for first_list, _ in pools]
    griddf_emb=pd.DataFrame(rows, columns=pool)
    #griddf_emb.to_csv('grid.csv')
    pools = pools[:-1] # The last one is embedding only (not for K>1 studies)
else:
    pools=[(CONFIG['embedding_vars'],CONFIG['segmentation_vars_categorical'])]
try:
    CONFIG.update({'n_latent_classes':int(STUDYNAME[-1])})
    print(f'Settings Configured. Number of LCs has been changed to {int(STUDYNAME[-1])} based on the STUDYNAME')
except ValueError:
    print('Basic setting has been configured; some will be done later in run_model.py')


#%% Main Optuna Experiments
import torch # placed here to reduce overhead for the above CONFIG definition
from src.run_model import run_model
torch.set_float32_matmul_precision('medium')

def objective(trial: optuna.trial.Trial, config:dict) -> float:
    """
    The Optuna objective function with dynamic epoch and batch size calculation.
    """
    # --- Define the Hyperparameter Search Space ---

    # 1. Determining epochs and batch size
    total_updates = trial.suggest_categorical("total_updates", [300, 600, 900, 1200, 1500])
    updates_per_epoch = trial.suggest_categorical("updates_per_epoch", [1, 10, 20, 40]) # 1: full-batch
    
    # Deterministically calculate epochs and batch size
    max_epochs = min(max(50,total_updates // updates_per_epoch),500)
    batch_size = max(128, math.ceil(config['train_size'] / updates_per_epoch))
    
    # 2. Optimizer and Regularization Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
    weight_decay_segmentation = trial.suggest_float("weight_decay_segmentation", 1e-4, 1e-2, log=True)
    weight_decay_embedding = trial.suggest_float("weight_decay_embedding", 1e-4, 1e-2, log=True)
    segmentation_dropout_rate = trial.suggest_float("segmentation_dropout_rate", 0.0, 0.4)

    # 3. Architectural Hyperparameters
    embedding_mode = trial.suggest_categorical("embedding_mode", ["shared", "class-specific"]) #maybe not optuna?
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 4)
    hidden_dims = [trial.suggest_categorical(f"n_nodes_layer_{i}", [64, 128, 256]) for i in range(num_hidden_layers)]

    # --- Update the global config with dynamically suggested hyperparameters ---
    config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "weight_decay_segmentation": weight_decay_segmentation,
        "weight_decay_embedding": weight_decay_embedding,
        "segmentation_dropout_rate": segmentation_dropout_rate,
        "embedding_mode": embedding_mode,
        "segmentation_hidden_dims": hidden_dims,
    })

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  - Derived params: epochs={max_epochs}, batch_size={batch_size}, hidden layer={hidden_dims}")

    # Optuna will try to MINIMIZE the value returned by this function.
    _, val_loss, final_metrics, _ = run_model(config=config, data2use=DATA2USE, verbose=False, logname=STUDYNAME)

    for key in ("test_acc", "test/rho2", "scaledentropy"):
        if key in final_metrics:
            trial.set_user_attr(key, final_metrics[key])

    return val_loss

# --- Set up Project Directory ---
STORAGE_NAME='sqlite:///optunaStudies.db'
#studies=optuna.get_all_study_summaries(storage=STORAGE_NAME)
#print([s.study_name for s in studies])
#optuna.delete_study(study_name='dcmseal2_SwissMetro', storage=STORAGE_NAME)
# --- Identify remaining gridsearch indices
if "explore" in STUDYNAME:
    matches=[re.compile(rf'{STUDYNAME}_(\d+)').match(f) for f in os.listdir()]
    remainings=set(range(len(pools)))-set([int(m.group(1)) for m in matches if m is not None])
else:
    remainings={0}
# --- Create and Run the Optuna Study ---
for i in remainings:
    tempstudyname=STUDYNAME+"_"+DATA2USE
    if "explore" in STUDYNAME:
        tempstudyname=tempstudyname+"_"+str(i)
    currentconfig=CONFIG.copy()
    currentconfig.update({'embedding_vars':pools[i][0],'segmentation_vars_categorical':pools[i][1]})
    print(currentconfig)
    tempstudyname=STUDYNAME+"_"+DATA2USE
    study = optuna.create_study(
        direction="minimize",
        study_name=tempstudyname,
        storage=STORAGE_NAME,
        load_if_exists=True
    )
    currentObj=partial(objective, config=currentconfig)
    if len(study.trials)<NUMTRIALS:
        study.optimize(currentObj, n_trials=NUMTRIALS-len(study.trials))
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    finalResults = study.trials_dataframe().dropna(subset=['value'])
    finalResults['duration']=finalResults['duration'].dt.total_seconds().astype(int)
    finalResults = finalResults.sort_values("value").reset_index(drop=True)
    finalResults.to_csv(STUDYNAME+"_"+str(i)+'_res.csv',index=False)
    #Organization
    keeplogs=finalResults.number[:LOGNUM]
    logfolder = Path.cwd() / "logs" / (DATA2USE + '_' + STUDYNAME)
    for p in logfolder.iterdir():
        if p.is_dir() and p.name.startswith("version_"):
            num = int(p.name.split("_")[1])
            if any(keeplogs == num):
                rank = str(1+int((keeplogs == num).idxmax())).zfill(2)   # idxmax returns the index of True
                p.rename(logfolder / f"{STUDYNAME}_{str(i)}_rank{rank}_{num}")
            else:
                shutil.rmtree(p)
    ckfolder=Path.cwd() / 'checkpoints'
    for p in ckfolder.iterdir():
        if p.is_file():
            if (DATA2USE+'_'+STUDYNAME) in p.name:
                os.remove(p)
    try:
        finalSegResults = finalResults.loc[(finalResults.user_attrs_scaledentropy>0.2) & (finalResults.user_attrs_scaledentropy<0.8),:]
        print(f'Finished: {round(len(finalSegResults)/len(finalResults),2)} of trials showed meaningful segmentation')
    except:
        print('Finished')

if '__file__' in globals():
    raise Exception('exploration has been completed; proceed with an IDE interactive env for post-search analyses')


#%% Post-Search Codes
if __name__ == "__main__":
    import torch
    from src.run_model import run_model
    from src.reporting_estimates import extract_betas, extract_embedding, predict_membership
    from scipy.optimize import linear_sum_assignment
    if (not DATA2USE) or (not CONFIG):
        raise Exception('Define DATA2USE and CONFIG first from the first block of main.py; currently only works for Synthesized')
    else:
        data2use=DATA2USE
        config=CONFIG.copy()

# ---- Store data for R mlogit (conventional MNLs) ----
if not (Path.cwd()/"mlogit"/"dfR.csv").exists() and DATA2USE=="Synthesized":
    from src.data_processing import load_and_preprocess_data
    data_dir = Path.cwd() / "data" / data2use
    train_df, test_df, _, _, _ = load_and_preprocess_data(config=config,data_dir=data_dir)
    def toR(df,config=config):
        core=config['core_vars']
        noncore=config['segmentation_vars_categorical']+config['embedding_vars']
        try:
            noncore=noncore+config["segmentation_vars_continuous"]
        except KeyError:
            pass
        if df.alt.min()==0:
            df['alt']=df['alt']+1
        df=df[['chid','alt','match']+core+list(df.filter(regex='^(' + '|'.join(noncore) + ')',axis=1).columns)].copy()
        def rename_columns(col_name):
            for prefix in noncore:
                if col_name.startswith(f"{prefix}_"):
                    # If a match is found, return the new name and stop checking
                    return col_name.replace(f"{prefix}_", "", 1)
            # If no prefix matches, return the original name
            return col_name
        dfLong=df.rename(columns=rename_columns)
        return dfLong
    dfTr=toR(train_df)
    dfTs=toR(test_df)
    dfTr['flag']='tr'
    dfTs['flag']='ts'
    dfFinal=pd.concat([dfTr,dfTs],ignore_index=True)
    dfFinal.to_csv('dfR.csv',index=False)

# ---- Reporting Functions ----
def calculate_mrae(row_true, row_estimated):
    """Calculate Mean Relative Absolute Error (MRAE) between two rows."""
    return np.mean(np.abs(row_true - row_estimated) / np.abs(row_true))

def match_rows(estimatedMat,mrscol='x1',exclEmbedding=True):
    """Match rows of estimatedMat to trueVal based on MRAE."""
    trueVal=pd.read_csv("data/Synthesized/trueMRS.csv",index_col="ind")
    trueVal=trueVal.copy().reset_index(drop=True)
    estimatedMatTemp=estimatedMat.copy().reset_index(drop=True)
    if exclEmbedding:
        lastCoreInd=np.where(trueVal.columns==config['core_vars'][-1])[0][0]+1
        trueVal=trueVal.iloc[:,:lastCoreInd].drop(columns=mrscol)
        estimatedMatTemp=estimatedMatTemp.iloc[:,:lastCoreInd].drop(columns=mrscol)
    mrae_matrix = np.zeros((len(trueVal), len(estimatedMatTemp)))    # Calculate the MRAE matrix
    for i, true_row in trueVal.iterrows():
        for j, estimated_row in estimatedMatTemp.iterrows():
            mrae_matrix[i, j] = calculate_mrae(true_row.values, estimated_row.values)
    row_indices, col_indices = linear_sum_assignment(mrae_matrix)    # Determine the optimal row matching using Hungarian Alg.
    return col_indices

def resReport(config,data2use=data2use,mrsdiv='x1'): #warning: if config is not fed, the pinned config upon the function definition will be used
    finalModel, bestobj, scores, dfInput = run_model(config,data2use,True)
    beta, beta_raw = extract_betas(finalModel, config)
    beta.index=['class_'+str(num+1) for num in beta.index]
    emb_df = extract_embedding(finalModel,config)
    def doEinsum(A=beta[config['embedding_vars']],C=emb_df):
        if len(C)==1: #copy global df across all classes
            B={K:C['global'] for K in A.index}
        else:
            B=C.copy()
        ref_key = list(A.index)[0]
        B_ref = B[ref_key]
        z_index = list(B_ref.index)
        j_cols  = list(B_ref.columns)
        e_labels = list(A.columns)
        e_to_idx = {e: i for i, e in enumerate(e_labels)}
        def _row_to_eidx(row_label: str) -> int:
            e_prefix = row_label.split('_', 1)[0]
            return e_to_idx[e_prefix]
        map_idx = np.array([_row_to_eidx(r) for r in z_index], dtype=int)  # (Z,)
        B_stack = np.stack([B[k].reindex(index=z_index, columns=j_cols).to_numpy() for k in list(A.index)],axis=0)  # (K, Z, J)
        A_values = A.to_numpy() # (K×E)
        W = A_values[:, map_idx]  # (K×Z), W[k,z] = A[k, E_of(z)]
        out = np.einsum('kz,kzj->kzj', W, B_stack) #(K×Z×J)
        result = out[:, :, 1:] - out[:, :, [0]] # Subtract the first column (broadcast across Z and K), then drop it
        new_cols = [f"{z}_{j}" for z in B_ref.index for j in B_ref.columns[1:]]
        flat = result.reshape(len(A), B_ref.shape[0] * (B_ref.shape[1]-1)) #to K×(Z*(J-1))
        df_flat = pd.DataFrame(flat, index=list(A.index), columns=new_cols)
        return df_flat
    if len(config['embedding_vars'])>0:
        emb_params=doEinsum()
        all_params=pd.concat([beta,emb_params],axis=1)
    else:
        all_params=beta
    mrs_df=all_params.div(all_params[mrsdiv].values,axis=0).drop(columns=config['embedding_vars'])
    membership_df=predict_membership(finalModel,dfInput).iloc[1::finalModel.hparams.n_alternatives] # start from 1 to facilitate post-hoc join
    emb_df = extract_embedding(finalModel,config)
    rhosq=scores['test/rho2']
    LL0=np.log(1/config['n_alternatives'])*config['test_size']*config['train_size']/config['n_alternatives'] # needs to be patched for homogeneous cases
    LLB=(1-rhosq)*LL0
    reorder=False
    if data2use=='Synthesized' and config['n_latent_classes']>1:
        print('Recalssifying the class order with the MRS-MRAE comparison with the ground truth')
        ordind=match_rows(mrs_df.copy())
        reorder=True
    if data2use=='SwissMetro': # reclassify based on VoT
        print('Reclassifying the class order with VoT; higher VoT=later class')
        ordind=list(mrs_df['TT'].rank().astype(int).values-1)
        reorder=True
    if reorder:
        print(f'Swapping the class order with {ordind}')
        originalLabel=all_params.index
        newLabel = originalLabel[ordind]
        all_params=all_params.set_index(newLabel).loc[originalLabel]
        mrs_df=mrs_df.set_index(newLabel).loc[originalLabel]
        membership_df.columns=newLabel
        membership_df=membership_df[originalLabel]
        emb_df=dict(zip(newLabel, emb_df.values()))
        emb_df={k: emb_df[k] for k in originalLabel if k in emb_df}
    classAssigned=np.argmax(membership_df.values,axis=1)
    _, counts = np.unique(classAssigned, return_counts=True)
    prop=counts/len(membership_df)
    results={'model':finalModel,'params':all_params,'mrs':mrs_df,'membership':membership_df,'embeddings':emb_df,'rho':rhosq,'LL':LLB,'CEL':bestobj,'memprop':prop}
    reportcols=list(all_params.columns[all_params.columns.str.contains('ASC_')])+config['core_vars']
    print(round(results['mrs'][reportcols],2))
    return results

#Individual studies
config=CONFIG.copy()
if data2use=="Synthesized":
    maxMRS=20
    store='mrs'
    mrsdiv='x1'
    storage='sqlite:///results/251020_Synthetic/251020_Synthesized.db'
    studies=optuna.get_all_study_summaries(storage=storage)
    print([s.study_name for s in studies])
    additUpdate={'segmentation_vars_categorical': ['s1', 's2', 'o1'], #safeguard
                 'embedding_vars': ['e1', 'e2']}
    match STUDYNAME:
        case "lccm":
            studyname='lccm'
            config=CONFIG.copy()
            config.update({
                "segmentation_hidden_dims": [],
                "embedding_mode": "class-specific", 
                "learning_rate": 0.02,
                "segmentation_dropout_rate": 0,
                "weight_decay_segmentation": 0,
                "weight_decay_embedding": 0,
                "batch_size": 999999,
                "max_epochs": 100,
            })
            rhocut=0.52 # no stats available; empirical cutoff to match similar # of req'd trials
        case "emnl":
            studyname="emnl_Synthesized_0"
            additUpdate={'segmentation_vars_categorical': [],'n_latent_classes':1,
                         'embedding_vars': ['e1', 'e2','s1', 's2', 'o1'],}
        case "nnlccm":
            studyname="nnlccm_Synthesized_0"
            additUpdate={'segmentation_vars_categorical': ['e1', 'e2','s1', 's2', 'o1'],
                         'embedding_vars': []}
        case "dcmseal":
            studyname="dcmseal_Synthesized_0"
    if studyname!="lccm":
        study = optuna.create_study(direction="minimize",study_name=studyname,storage=storage,load_if_exists=True)
        dfStudy=study.trials_dataframe().dropna(subset=['value']).sort_values("value")
        trial = next(t for t in study.get_trials(deepcopy=False) if t.number == dfStudy.iloc[0,0])
        config.update(trial.params)
        config['max_epochs'] = min(max(50,config['total_updates'] // config['updates_per_epoch']),500)
        config['batch_size'] = max(128, math.ceil(config['train_size'] / config['updates_per_epoch']))
        config["segmentation_hidden_dims"]=[config[f'n_nodes_layer_{i}'] for i in range(config['num_hidden_layers'])]
        rhocut=dfStudy.iloc[9,:]['user_attrs_test/rho2'] # top 5%
    config.update(additUpdate)

if data2use=="SwissMetro":
    altdef={0:"Train",1:"SM",2:"Car"}
    maxMRS=60
    store='params'
    mrsdiv='CO'
    storage='sqlite:///results/251026_SwissMetro/SwissMetro.db'
    studies=optuna.get_all_study_summaries(storage=storage)
    print([s.study_name for s in studies])
    allvars=config['embedding_vars']+config['segmentation_vars_categorical']
    if 'nnlccm' in STUDYNAME:
        additUpdate={'segmentation_vars_categorical': allvars,
                     'embedding_vars': [],'n_latent_classes':int(STUDYNAME[-1])}
    elif 'emnl' in STUDYNAME:
        additUpdate={'segmentation_vars_categorical': [],'n_latent_classes':1,
                     'embedding_vars':allvars}
    elif 'dcmseal' in STUDYNAME:
        additUpdate={'n_latent_classes':int(STUDYNAME[-1])}
    studyname=STUDYNAME+'_'+data2use
    study = optuna.create_study(direction="minimize",study_name=studyname,storage=storage,load_if_exists=True)
    dfStudy=study.trials_dataframe().dropna(subset=['value']).sort_values("value")
    trial = next(t for t in study.get_trials(deepcopy=False) if t.number == dfStudy.iloc[0,0])
    config.update(trial.params)
    config['max_epochs'] = min(max(50,config['total_updates'] // config['updates_per_epoch']),500)
    config['batch_size'] = max(128, math.ceil(config['train_size'] / config['updates_per_epoch']))
    config["segmentation_hidden_dims"]=[config[f'n_nodes_layer_{i}'] for i in range(config['num_hidden_layers'])]
    rhocut=dfStudy.iloc[9,:]['user_attrs_test/rho2'] # top 5%
    config.update(additUpdate)

def saveResults(numStat=30, maxMRS=20,store='params',rhocut=rhocut, savename=STUDYNAME):
    modelcount=0
    goodcount=0
    while goodcount<numStat:
        res=resReport(config,mrsdiv=mrsdiv)
        modelcount+=1
        if res['rho']>rhocut and res['mrs'].abs().max().max()<maxMRS: 
            goodcount+=1
            betanames=np.array([f'{x}_{y}' for y in [i for i in res[store].index] for x in res[store].columns])
            storeit=pd.Series(np.append(np.array(res['rho']),np.array(res[store]).flatten()),
                              index=np.append(np.array('rho_ts'),betanames))
            storeit['LL']=res['LL']
            storeit['CEL']=res['CEL']
            if res['model'].hparams.n_latent_classes>1:
                storeit['c1prop']=res['memprop'][0]
            if goodcount==1:
                dataOut=pd.DataFrame({('Model_'+str(modelcount)):storeit})
            else:
                dataOut[('Model_'+str(modelcount))]=storeit
                dataOut=dataOut.copy()
            if data2use=="SwissMetro" and 'emnl' not in studyname:
                res['membership'].to_csv('results/member_'+STUDYNAME+'_'+str(modelcount)+'.csv')
                if len(res['model'].hparams["embedding_vars"])>0:
                    embs=pd.concat(res['embeddings'].values(), axis=0)
                    embs.to_csv('results/emb_'+STUDYNAME+'_'+str(modelcount)+'.csv')
                    torch.save(res['model'], 'results/model_'+STUDYNAME+'_'+str(modelcount)+'.pth')
            global dfTemp
            dfTemp=dataOut.T.copy()
            print('***********This run has been recorded************')
    dataOut.T.to_csv('results/res_'+data2use+'_'+savename+'.csv')
    return dataOut.T


if not (Path.cwd()/"results"/('res_'+data2use+'_'+STUDYNAME+'.csv')).exists():
    saveResults(maxMRS=maxMRS,store=store,rhocut=rhocut)

