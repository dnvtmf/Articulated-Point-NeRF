_base_ = './default.py'

expname = 'pandas'
basedir = './logs/wim_50/'

data = dict(
    datadir='./data/WIM/pandas',
    dataset_type='wim',
    canonical_t=0.96,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
