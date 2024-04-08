_base_ = './default.py'

expname = f'393'
basedir = './logs/zju_new/'

data = dict(
    datadir='./data/zju/393/cache_train.pickle',
    dataset_type='zju',
    # Training data
    inverse_y=True,
    canonical_t=0.,
    video_len=658,
    flip_x=False,
    flip_y=False,
)
