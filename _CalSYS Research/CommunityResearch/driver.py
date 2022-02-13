
import pandas as pd
import networkx as nx
from Utils.networks import make_net, save_net, show_net, get_net
from earlyadopters import get_early_adopters
from get_ruicheng_features import get_rc_features
from get_original_features import get_og_features

from Utils.balance import prepare
from Utils.dataset_split import spltdataset
from model_driver import train_models

#forums = [34,41,77,84] # identified topics
mult = [10,15,20,30,50]
alpha = 7
forums = [41]#[34,41,77,84]

research = ['original', 'ruicheng'] # 'ruicheng'

for forum in forums:
    for rs in research:
        acc_avg, f1_avg = [], [] # empty
        for m in mult:
            print(f'Doing Forum {forum} with {m}X using {rs}')
            pkp = f'DirectedNetworks\pickleX{forum}.p'
            acc,f1 = [], []
            # run query to retrieve topics id's where above > size and > size/2
            csc, ncsc, tcsc, tncsc = get_early_adopters(forum, alpha, alpha*m) # set threshold here    
            
            for n in range(2):
                # Prepare sets
                split = 0.8 # not over 1
                train_threads, train_times, test_threads, test_times = prepare(csc, ncsc, tcsc, tncsc, split, 2) # 1 = both balanced; 2 = train balanced, test imbalanced; 3 = both imbalanced
                
                if rs == 'ruicheng':
                    # Get features -> save as pdf
                    train_df = get_rc_features(train_threads, train_times, get_net(pkp))
                    
                    # Get features -> save as pdf
                    test_df = get_rc_features(test_threads, test_times, get_net(pkp))
                
                elif rs == 'original':
                    # Get features -> save as pdf
                    train_df = get_og_features(train_threads, train_times, get_net(pkp))
                    
                    # Get features -> save as pdf
                    test_df = get_og_features(test_threads, test_times, get_net(pkp))
                else:
                    print('Please select which features to use!')
                    continue

                # Split dataset
                X_test, X_train, Y_test, Y_train = spltdataset(train_df, test_df)
                
                # Train Models
                accuracy, f1_score = train_models(X_test, X_train, Y_test, Y_train, forum, m, ['RF_ADA'])
                acc.append(accuracy)
                f1.append(f1_score)
        
            acc_avg.append(sum(acc)/len(acc))
            f1_avg.append(sum(f1)/len(f1))

        for i in range(len(mult)):
            print(f'M: {mult[i]}X Acc: {acc_avg[i]} F1: {f1_avg[i]}')