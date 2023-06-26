import argparse
import pickle
import os
import time
from functions import generate_dir2save

def read_arguments(train=True):
    parser = get_arguments()
    opt = parser.parse_args()
    opt = parser.parse_args()
    opt.device = "cpu" if opt.cpu else "cuda:0"
    opt.phase = 'train' if train else 'test'
    if train:
        opt.continue_epoch = 0 if not opt.continue_train else load_iter(opt)
    else:
        opt.continue_epoch = opt.which_epoch
    if train:
        save_options(opt, parser)
    return opt


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--dataroot', help='location of datasets', default='datasets/')
    parser.add_argument('--dataset_name', help='dataset name', default='Estaillades carbonate_80') 
    parser.add_argument('--num_epochs', type=int, default=60000, help='number of epochs') 
    parser.add_argument('--max_size', type=int, help='limit image size in max dimension', default=80) 
    parser.add_argument('--continue_train', action="store_true", help='continue training of a previous checkpoint?')
    parser.add_argument('--which_epoch', type=int, help='which epoch to use for evaluation')
    parser.add_argument('--num_generated', type=int, default=1, help='which epoch to use for evaluation')
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)

    parser.add_argument('--noise_amp', type=float, help='additive noise cont weight', default=0.1)
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # regime
    parser.add_argument('--no_masks', action='store_true', help='use the regime without segmentation masks')
    parser.add_argument('--use_kornia_augm', action='store_true', help='use an older version of differentiable augm')

    # training:
    parser.add_argument('--batch_size', type=int, help='batch_size', default=3)
    parser.add_argument('--noise_dim', type=int, help='dimension of noise vector', default=64)
    parser.add_argument('--lr_g', type=float, default=0.0004, help='generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0004, help='discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--loss_mode', help='which GAN loss (wgan|hinge|bce)', default="bce")
    parser.add_argument('--seed', type=int, help='which randomm seed to use', default=22)
    parser.add_argument('--no_DR', action="store_true", help='deactivate Diversity Regularization?')
    parser.add_argument('--prob_augm', type=float, help='probability of data augmentation', default=0.3)
    parser.add_argument('--lambda_DR', type=float, help='lambda for DR', default=0.15)
    parser.add_argument('--no_EMA', action="store_true", help='deactivate exponential moving average of G weights?')
    parser.add_argument('--EMA_decay', type=float, help='decay for exponential moving averages', default=0.8)
    parser.add_argument('--bernoulli_warmup', type=int, help='epochs for soft_mask bernoulli warmup', default=15000)
    parser.add_argument('--lambda_grad', type=float, help='gradient penalty weight', default=0.1)  

    # architecture
    parser.add_argument('--norm_G', help='which norm to use in generator     (None|batch|instance)', default="none")
    parser.add_argument('--norm_D', help='which norm to use in discriminator (None|batch|instance)', default="none")
    parser.add_argument('--ch_G', type=float, help='channel multiplier for G blocks', default=8)
    parser.add_argument('--ch_D', type=float, help='channel multiplier for D blocks', default=8)
    parser.add_argument('--num_blocks_d', type=int, help='Discriminator blocks number. 0 -> use recommended default', default=0)
    parser.add_argument('--num_blocks_d0', type=int, help='Num of D_low-level blocks. 0 -> use recommended default', default=0)

    # stats tracking
    parser.add_argument('--freq_save_loss', type=int, help='frequency of loss plot updates', default=200)
    parser.add_argument('--freq_print', type=int, help='frequency of saving images and timer', default=200)
    parser.add_argument('--freq_save_ckpt', type=int, help='frequency of saving checkpoints', default=200)

    return parser


def update_options_from_file(opt, parser):
    opt.dir = generate_dir2save(opt)
    file_name = '%s/opt.pkl' % (opt.dir)
    new_opt = pickle.load(open(file_name, 'rb'))
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_iter(opt):
    opt.dir = generate_dir2save(opt)
    with open('%s/models/latest_epoch.txt' % (opt.dir), "r") as f:
        res = int(f.read())
        return res

def save_options(opt, parser):
    opt.dir = generate_dir2save(opt)
    path_name = '%s' % (opt.dir)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'at') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'ab') as opt_file:
        pickle.dump(opt, opt_file)

