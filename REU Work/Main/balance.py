import math

# x = prepare(csc, ncsc, tcsc, tncsc, split, 2) # 1 = both balanced; 2 = tain balanced, test imbalanced; 3 = both imbalanced#
def prepare(pos, neg, tpos, tneg, split, case_id):
    test_threads = {}
    test_threads_pos = {}
    test_threads_neg = {}

    test_times = {}
    test_times_pos = {}
    test_times_neg = {}

    train_threads = {}
    train_threads_pos = {}
    train_threads_neg = {}

    train_times = {}
    train_times_pos = {}
    train_times_neg = {}

    pos_threads = list(pos.keys())
    neg_threads = list(neg.keys())

    # assume neg >> pos
    ratio = math.floor(len(neg) / len(pos))
    print(f'ratio is {ratio}')

    cnt = math.floor(len(pos_threads)*split) # 80%
    cnt2 = len(pos_threads) - cnt
    print(f'train count should be {cnt*2}')
    print(f'test count should be {cnt2 + cnt2*ratio}')
    print(f'{ratio*cnt2} for {cnt2}')

    for i in range(len(pos_threads)):
        # Training Set
        if i < cnt:
            train_threads_neg[neg_threads[i]] = neg[neg_threads[i]] # get users per id
            train_threads_pos[pos_threads[i]] = pos[pos_threads[i]]
            train_times_neg[neg_threads[i]] = tneg[neg_threads[i]]
            train_times_pos[pos_threads[i]] = tpos[pos_threads[i]]
        
        # Testing Set
        else:
            test_threads_pos[pos_threads[i]] = pos[pos_threads[i]]
            test_times_pos[pos_threads[i]] = tpos[pos_threads[i]]
            for j in range(1,ratio+1):
                test_threads_neg[neg_threads[i+cnt2*j]] = neg[neg_threads[i+cnt2*j]] # get users per id
                test_times_neg[neg_threads[i+cnt2*j]] = tneg[neg_threads[i+cnt2*j]]

    # Combine
    train_threads['Pos'] = train_threads_pos
    train_threads['Neg'] = train_threads_neg

    train_times['Pos'] = train_times_pos
    train_times['Neg'] = train_times_neg

    test_threads['Pos'] = test_threads_pos
    test_threads['Neg'] = test_threads_neg

    test_times['Pos'] = test_times_pos
    test_times['Neg'] = test_times_neg

    return train_threads, train_times, test_threads, test_times

    ''' # down sampling
    # check for average 
    # try balanced vs unbalanced dataset
    dwns = sorted(ns, key=ns.get, reverse=False)[:len(casc)] # get biggest noncasc id's
    for d in dwns:
        noncasc_d[d] = noncasc[d]
        timencsc_d[d] = timencsc[d]
        avg.append((nonsize[d]))

    print(f'average is {sum(avg)/len(avg)} where {len(qdf2)} becomes {len(qdf)}')
    '''
    '''diff = len(noncasc) - len(casc)
    for key in rd.sample(noncasc.keys(), diff):
        del noncasc[key]
        del timencsc[key]'''

# 80/20  --> 31/8

