import datetime
import numpy as np
import os
import pandas as pd
from scipy.sparse import coo_matrix
import sys
#import sh
import tensorflow as tf

import wals

# ratio of train set size to test set size
TEST_SET_RATIO = 10



def make_tran_and_test(input_file):

    headers = ['user_id', 'item_id', 'rating', 'timestamp']
    header_row = 0 if False else None
    ratings_df = pd.read_csv(input_file,
                            sep='\t',
                            names=headers,
                            header=header_row,
                            dtype={
                                'user_id': np.int32,
                                'item_id': np.int32,
                                'rating': np.float32,
                                'timestamp': np.int32,
                            })

    np_users = ratings_df.user_id.as_matrix()
    np_items = ratings_df.item_id.as_matrix()
    

    unique_users = np.unique(np_users)
    unique_items = np.unique(np_items)

    n_users = unique_users.shape[0]
    n_items = unique_items.shape[0]

    print(n_users)


    # make indexes for users and items if necessary
    max_user = unique_users[-1]
    max_item = unique_items[-1]

    print(max_user)
    if n_users != max_user or n_items != max_item:
        # make an array of 0-indexed unique user ids corresponding to the dataset
        # stack of user ids
        z = np.zeros(max_user+1, dtype=int)
        print (z)
        z[unique_users] = np.arange(n_users)
        u_r = z[np_users]

        # make an array of 0-indexed unique item ids corresponding to the dataset
        # stack of item ids
        z = np.zeros(max_item+1, dtype=int)
        z[unique_items] = np.arange(n_items)
        i_r = z[np_items]

        # construct the ratings set from the three stacks
        np_ratings = ratings_df.rating.as_matrix()
        ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
        ratings[:, 0] = u_r
        ratings[:, 1] = i_r
        ratings[:, 2] = np_ratings
    else:
        ratings = ratings_df.as_matrix(['user_id', 'item_id', 'rating'])
        #print(ratings)
        # deal with 1-based user indices
        ratings[:, 0] -= 1
        ratings[:, 1] -= 1

    tr_sparse, test_sparse = _create_sparse_train_and_test(ratings,
                                                            n_users, n_items)

    return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse



def _create_sparse_train_and_test(ratings, n_users, n_items):
    """Given ratings, create sparse matrices for train and test sets.

    Args:
        ratings:  list of ratings tuples  (u, i, r)
        n_users:  number of users
        n_items:  number of items

    Returns:
        train, test sparse matrices in scipy coo_matrix format.
    """
    # pick a random test set of entries, sorted ascending
    test_set_size = len(ratings) / TEST_SET_RATIO
    test_set_idx = np.random.choice(xrange(len(ratings)),
                                    size=test_set_size, replace=False)
    test_set_idx = sorted(test_set_idx)

    # sift ratings into train and test sets
    ts_ratings = ratings[test_set_idx]
    tr_ratings = np.delete(ratings, test_set_idx, axis=0)
    print(tr_ratings.shape)

    # create training and test matrices as coo_matrix's
    u_tr, i_tr, r_tr = zip(*tr_ratings)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

    u_ts, i_ts, r_ts = zip(*ts_ratings)    
    test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))

    return tr_sparse, test_sparse



def train_model(tr_sparse):
    """Instantiate WALS model and use "simple_train" to factorize the matrix.

    Args:
        args: training args containing hyperparams
        tr_sparse: sparse training matrix

    Returns:
        the row and column factors in numpy format.
    """
    dim = 5
    num_iters = 20
    reg = 0.07
    unobs = 0.01
    wt_type = 0
    feature_wt_exp = 0.08
    obs_wt = 130

    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # generate model
    input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse,
                                                                    dim,
                                                                    reg,
                                                                    unobs,
                                                                    True,
                                                                    wt_type,
                                                                    feature_wt_exp,
                                                                    obs_wt)

    # factorize matrix
    session = wals.simple_train(model, input_tensor, num_iters)

    print('end')
    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)

    # close the training session now that we've evaluated the output
    session.close()

    return output_row, output_col



def main():
    user_map, item_map, tr_sparse, test_sparse = make_tran_and_test(sys.argv[1])

    output_row, output_col = train_model(tr_sparse)

    train_rmse = wals.get_rmse(output_row, output_col, tr_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)

    print(train_rmse)
    print(test_rmse)

main()