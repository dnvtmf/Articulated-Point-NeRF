_base_ = './default.py'

expname = 'cassie'
basedir = './logs/wim/'

data = dict(
    datadir='./data/WIM/cassie',
    dataset_type='wim',
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
