_base_ = './default.py'

expname = 'spot'
basedir = './logs/wim_50/'

data = dict(
    datadir='./data/WIM/spot',
    dataset_type='wim',
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
