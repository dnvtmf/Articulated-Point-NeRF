_base_ = './default.py'

expname = 'atlas'
basedir = './logs/wim_50/'

data = dict(
    datadir='./data/WIM/atlas',
    dataset_type='wim',
    canonical_t=0.03,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
