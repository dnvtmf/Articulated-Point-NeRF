_base_ = './default.py'

expname = 'nao'
basedir = './logs/wim_50/'

data = dict(
    datadir='./data/WIM/nao',
    dataset_type='wim',
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
