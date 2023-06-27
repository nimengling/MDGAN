import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm #as sp_norm
import copy
from .utils import to_rgb, from_rgb, to_decision, get_norm_by_name
from .feature_augmentation import Content_FA, Layout_FA
from functions import generate_dir2save


def create_models(opt, recommended_config):
    """
    Build the model configurations and create models
    """
    config_G, config_D = prepare_config(opt, recommended_config)

    # --- generator and EMA --- #
    netG = Generator(config_G).to(opt.device)
    netG.apply(weights_init)
    netEMA = copy.deepcopy(netG) if not opt.no_EMA else None 

    # --- discriminator --- #
    if opt.phase == "train":
        netD = Discriminator(config_D).to(opt.device)
        netD.apply(weights_init)
    else:
        netD = None

    # --- load previous ckpt  --- #
    opt.dir = generate_dir2save(opt)
    path = '%s/models' % (opt.dir)
     if opt.continue_train or opt.phase == "test":
        netG.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch)+"_G.pth")))
        print("Loaded Generator checkpoint")
        if not opt.no_EMA:
            netEMA.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch)+"_G_EMA.pth")))
            print("Loaded Generator_EMA checkpoint")
    if opt.continue_train and opt.phase == "train":
        netD.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch)+"_D.pth")))
        print("Loaded Discriminator checkpoint")
    return netG, netD, netEMA


def create_optimizers(netG, netD, opt):
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
    return optimizerG, optimizerD


def prepare_config(opt, recommended_config):
    """
    Create model configuration dicts based on recommended settings and input parameters.
    Recommended num_blocks_d and num_blocks_d0 can be overridden by user inputs
    """
    G_keys_recommended = ['noise_shape', 'num_blocks_g', "no_masks", "num_mask_channels"]
    D_keys_recommended = ['num_blocks_d', 'num_blocks_d0', "no_masks", "num_mask_channels"]
    G_keys_user = ["ch_G", "norm_G", "noise_dim"]
    D_keys_user = ["ch_D", "norm_D", "bernoulli_warmup"]

    config_G = dict((k, recommended_config[k]) for k in G_keys_recommended)###recommended_config是一个字典,将recommended_config这个字典中的包含G_keys_recommended的键值的键值对保存在config_G之中
    config_G.update(dict((k, getattr(opt, k)) for k in G_keys_user))
    config_D = dict((k, recommended_config[k]) for k in D_keys_recommended)
    config_D.update(dict((k, getattr(opt, k)) for k in D_keys_user))

    if opt.num_blocks_d > 0:
        config_D["num_blocks_d"] = opt.num_blocks_d
    if opt.num_blocks_d0 > 0:
        config_D["num_blocks_d0"] = opt.num_blocks_d0
    return config_G, config_D



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_channels(which_net, base_multipler):
    channel_multipliers = {
        "Generator": [8, 8, 8, 8, 8, 8, 8, 4, 2, 1],
        "Discriminator": [1, 2, 4, 8, 8, 8, 8, 8, 8]
    }
    ans = list()
    for item in channel_multipliers[which_net]:
        ans.append(int(item * base_multipler))
    return ans


class Generator(nn.Module):
    def __init__(self, config_G):
        super(Generator, self).__init__()
        self.num_blocks = config_G["num_blocks_g"]
        self.noise_shape = config_G["noise_shape"]
        self.noise_init_dim = config_G["noise_dim"]
        self.norm_name = config_G["norm_G"]
        self.no_masks = config_G["no_masks"]
        self.num_mask_channels = config_G["num_mask_channels"]
        num_of_channels = get_channels("Generator", config_G["ch_G"])[-self.num_blocks-1:]# [-6:]取"Generator": [8, 8, 8, 8, 8, 8, 8, 4, 2, 1]的右边6个和ch_G数相乘
        self.body, self.rgb_converters = nn.ModuleList([]), nn.ModuleList([])
        self.first_linear = nn.ConvTranspose3d(self.noise_init_dim, num_of_channels[0], self.noise_shape)#nn.ConvTranspose3d(64,64,（5,5,5)），kernel_size=（5,5,5),out_put=(3,64,d,h,w)
        for i in range(self.num_blocks):
            cur_block = G_block(num_of_channels[i], num_of_channels[i+1], self.norm_name, i==0)#(64,64,None,i==0)
            cur_rgb   = to_rgb(num_of_channels[i+1])
            self.body.append(cur_block)
            self.rgb_converters.append(cur_rgb)
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        print("Created Generator with %d parameters" % (sum(p.numel() for p in self.parameters())))

    def generate(self, z, get_feat=False):
        output = dict()
        ans_images = list()
        ans_feat = list()
        x = self.first_linear(z) #z:torch.Size([3, 64, 1, 1, 1]) x:torch.Size([3, 64, 5, 5, 5])
        for i in range(self.num_blocks):
            x = self.body[i](x)  #x torch.Size([3, 64, 5, 5, 5])
            im = torch.tanh(self.rgb_converters[i](x)) #im torch.Size([3, 3, 5, 5, 5])
            ans_images.append(im)  #(3, 3, 5, 5, 5）
            ans_feat.append(torch.tanh(x))
        output["images"] = ans_images

        if get_feat:
             output["features"] = ans_feat
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        return output


class G_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first):#(64,64,None,i==0)
        super(G_block, self).__init__()
        middle_channel = min(in_channel, out_channel)
        self.ups = nn.Upsample(scale_factor=2) if not is_first else torch.nn.Identity() #mode='trilinear'
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = torch.nn.utils.spectral_norm (nn.Conv3d(in_channel, middle_channel, (3, 3, 3), padding=1))
        self.conv2 = torch.nn.utils.spectral_norm (nn.Conv3d(middle_channel, out_channel, (3,3,3), padding=1))
        self.norm1  = get_norm_by_name(norm_name, in_channel)
        self.norm2  = get_norm_by_name(norm_name, middle_channel)
        self.conv_sc = torch.nn.utils.spectral_norm (nn.Conv3d(in_channel, out_channel, (1, 1, 1), bias=False))

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.activ(x)
        x = self.ups(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        h = self.ups(h)
        h = self.conv_sc(h)
        return h + x


class Discriminator(nn.Module):
    def __init__(self, config_D):
        super(Discriminator, self).__init__()
        self.num_blocks = config_D["num_blocks_d"]
        self.num_blocks_ll = config_D["num_blocks_d0"]
        self.norm_name = config_D["norm_D"]
        self.no_masks = config_D["no_masks"]
        self.num_mask_channels = config_D["num_mask_channels"]
        self.bernoulli_warmup = config_D["bernoulli_warmup"]
        num_of_channels = get_channels("Discriminator", config_D["ch_D"])[:self.num_blocks + 1]#[8,16,32,64, 64, 64,64]
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        self.feature_prev_ratio = 8  # for msg concatenation

        self.body_ll, self.body_content, self.body_layout = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        self.rgb_to_features = nn.ModuleList([])  # for msg concatenation
        self.final_ll, self.final_content, self.final_layout = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])

        # --- D low-level --- #
        for i in range(self.num_blocks_ll): #i=0,1
            msg_channels = num_of_channels[i] // self.feature_prev_ratio if i > 0 else num_of_channels[0] #i=0时为8,i=1时为2
            in_channels = num_of_channels[i] + msg_channels if i > 0 else num_of_channels[0]  #i=0时为8,i=1时为18
            cur_block = D_block(in_channels, num_of_channels[i+1], self.norm_name, is_first=i == 0) ##i=0时D_block(8,16,none,..),i=1时D_block(18,32,none,..)
            self.body_ll.append(cur_block)
      
        # --- D content --- #
        for i in range(self.num_blocks_ll, self.num_blocks): #(2,6),i=2,3,4,5
            k = i - self.num_blocks_ll
            cur_block_content = D_block(num_of_channels[i], num_of_channels[i + 1], self.norm_name, only_content=True)
            self.body_content.append(cur_block_content)
            out_channels = 1 if self.no_masks else self.num_mask_channels + 1
            self.final_content.append(to_decision(num_of_channels[i + 1], out_channels))

        # --- D layout --- #
        for i in range(self.num_blocks_ll, self.num_blocks): #(2,6),i=2,3,4,5
            k = i - self.num_blocks_ll  #0,1,2,3
            in_channels = 1 if k > 0 else num_of_channels[i]
            cur_block_layout = D_block(in_channels, 1, self.norm_name) #(8,1,)
            self.body_layout.append(cur_block_layout)
            self.final_layout.append(to_decision(1, 1))
        print("Created Discriminator (%d+%d blocks) with %d parameters" %
              (self.num_blocks_ll, self.num_blocks-self.num_blocks_ll, sum(p.numel() for p in self.parameters())))

    def discriminate(self, inputs, for_real, epoch):
        images = inputs["images"]
        output_ll, output_content, output_layout = list(), list(), list(),

        # --- D low-level --- #
        y = self.rgb_to_features[0](images[-1])
        for i in range(0, self.num_blocks_ll): #(0,1)
            if i > 0:
                y = torch.cat((y, self.rgb_to_features[i](images[-i - 1])), dim=1)
                #print('y2.type', type(y)) #torch.Tentor
            y = self.body_ll[i](y)
            output_ll.append(self.final_ll[i](y))
        
        # --- D content --- #
        y_con = y
        y_con = torch.mean(y_con, dim=(2, 3, 4), keepdim=True)
        for i in range(self.num_blocks_ll, self.num_blocks):  #(2,6)
            k = i - self.num_blocks_ll  #0,1,2,3
            y_con = self.body_content[k](y_con)
            output_content.append(self.final_content[k](y_con))

        # --- D layout --- #
        y_lay = y
        for i in range(self.num_blocks_ll, self.num_blocks): #(2,6)
            k = i - self.num_blocks_ll  #0,1,2,3
            y_lay = self.body_layout[k](y_lay)
            output_layout.append(self.final_layout[k](y_lay))

        return {"low-level": output_ll, "content": output_content, "layout": output_layout}


class D_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first=False, only_content=False):
        super(D_block, self).__init__()
        middle_channel = min(in_channel, out_channel) #8;18
        ker_size, padd_size = (1, 0) if only_content else (3, 1)
        self.is_first = is_first
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = torch.nn.utils.spectral_norm (nn.Conv3d(in_channel, middle_channel, ker_size, padding=padd_size))
        self.conv2 = torch.nn.utils.spectral_norm (nn.Conv3d(middle_channel, out_channel, ker_size, padding=padd_size))
        self.norm1 = get_norm_by_name(norm_name, in_channel)
        self.norm2 = get_norm_by_name(norm_name, middle_channel)
        self.down = nn.AvgPool3d(2) if not only_content else torch.nn.Identity() #AvgPool3d(kernel_size=2, stride=2, padding=0)
        learned_sc = in_channel != out_channel or not only_content
        print('learned_sc',learned_sc)
        if learned_sc:
            print('in_channel, out_channel,',in_channel, out_channel)
            self.conv_sc = torch.nn.utils.spectral_norm (nn.Conv3d(in_channel, out_channel, (1, 1, 1), bias=False))
        else:
            self.conv_sc = torch.nn.Identity()

    def forward(self, x):
        h = x
        print('1',not self.is_first)
        if not self.is_first:
            x = self.norm1(x)
            x = self.activ(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        if not x.shape[0] == 0:
            x = self.down(x)
        h = self.conv_sc(h)
        if not x.shape[0] == 0:
            h = self.down(h)
        return x + h
