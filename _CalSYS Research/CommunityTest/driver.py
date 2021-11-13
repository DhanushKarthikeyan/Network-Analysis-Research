
import pandas as pd
import networkx as nx
from networks import make_net, save_net, show_net, get_net
from earlyadopters import get_early_adopters
#from feature_extraction import get_features
from get_comm_features import get_features
from balance import prepare

#forums = [34,41,77,84] # identified topics
mult = [2,3]
alpha = 25
for m in mult:
    forums = [77]#[34,41,77,84]
    for forum in forums:
        print(f'Doing Forum {forum}')
        pkp = f'pickleX{forum}.p'

        '''
        net = make_net(forum)
        pkp = save_net(net, forum)
        '''
        # show_net(pkp, forum)

        # run query to retrieve topics id's where above > size and > size/2
        csc, ncsc, tcsc, tncsc = get_early_adopters(forum, alpha, alpha*m) # set threshold here    
        
        # Prepare sets
        split = 0.8 # not over 1
        train_threads, train_times, test_threads, test_times = prepare(csc, ncsc, tcsc, tncsc, split, 2) # 1 = both balanced; 2 = train balanced, test imbalanced; 3 = both imbalanced
        
        # Get features -> save as pdf
        train_df = get_features(train_threads, train_times, get_net(pkp))
        train_df.to_csv(f'Forum{forum}data_{m}x_train.csv', header = True, index = False)
        del train_df
        
        # Get features -> save as pdf
        test_df = get_features(test_threads, test_times, get_net(pkp))
        test_df.to_csv(f'Forum{forum}data_{m}x_test.csv', header = True, index = False)
        del test_df
