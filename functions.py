import torch
import numpy as np
import datetime
import dateutil.tz
import config

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:,:]) 
        inp = inp.numpy().transpose((1,2,3,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:,:])
        inp = inp.numpy().transpose((0,1,2))
    inp = np.clip(inp,0,1)
    return inp

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def generate_dir2save(opt):
    dir2save = 'checkpoints/{}/'.format(opt.dataset_name)
    dir2save += datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    dir2save += "_batch_size:{}".format(opt.batch_size)
    dir2save += "_loss_mode:{}".format(opt.loss_mode)
    dir2save += "_lr_g_{}_lr_d_{}".format(opt.lr_g, opt.lr_d)
    dir2save += "_EMA_decay_{}".format(opt.EMA_decay)
    return dir2save

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    gradient_penalty = torch.tensor([])
    for i in range(5):

        real_data = torch.tensor(real_data['images'][i])
        fake_data = torch.tensor(fake_data['images'][i])

        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                          grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                              #disc_interpolates.size()),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        #LAMBDA = 1
        gradient_penalty = gradient_penalty + ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty