from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from models.loss_ssim import SSIMLoss
from utils.net_mode import *
from utils.util import *

from DataFidelities.IDTClass import *
import torch.autograd as autograd

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def tensor_normlize(n_ipt:torch.tensor):
    if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
        dim_x, dim_y = n_ipt.shape[2:4]
        n_ipt_max = torch.max(n_ipt.max(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
        n_ipt_min = torch.min(n_ipt.min(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
        n_ipt_norm = (n_ipt - n_ipt_min) / (n_ipt_max - n_ipt_min)
    return n_ipt_norm, n_ipt_max, n_ipt_min

def tensor_denormlize(n_ipt, n_ipt_max, n_ipt_min):
    if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
        dim_x, dim_y = n_ipt.shape[2:4]
        n_ipt_denorm = torch.mul(n_ipt, n_ipt_max-n_ipt_min) + n_ipt_min
    return n_ipt_denorm

def anderson_img(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W, CX = x0.shape
    X = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break

        # if k%10==0:
        #     n_ipt_temp = torch.unbind(iFFT(X[:,k%m].view_as(x0)), -1)[0]
        #     n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)

    return X[:,k%m].view_as(x0), res, None#n_ipt_periter

def anderson_grad(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W, CX = x0.shape
    X = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break

    return X[:,k%m].view_as(x0), res

class URED(nn.Module):

    def __init__(self, dnn: nn.Module, config:dict, batch_size=40):
        #gamma_inti=3e-3, tau_inti=1e-1, batch_size=60, device='cuda:0'):

        super(URED, self).__init__()
        self.dnn = dnn

        self.gamma = torch.tensor(config['gamma_inti'], dtype=torch.float32)
        self.tau = torch.tensor(config['tau_inti'], dtype=torch.float32) # torch.nn.Parameter(torch.tensor(config['tau_inti'], dtype=torch.float32, requires_grad=False))#
        self.dObj = IDTClass()

    def denoise(self, n_ipt, create_graph=True, strict=True):
        n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
        n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
        denoiser = self.dnn(n_ipt_real, create_graph=create_graph, strict=create_graph)        
        denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
        denoiser = FFT(denoiser)     
        return n_ipt - denoiser

    def forward(self, n_ipt, n_y=None, emParams=None, meas_list=None, gt=None, create_graph=True, strict=True):
        delta_g = self.dObj.fgrad_SGD(n_ipt, n_y, meas_list, emParams)
        n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
        n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
        denoiser = self.dnn(n_ipt_real, create_graph, strict)        
        denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
        denoiser = FFT(denoiser)
        xSubD    = self.tau * (n_ipt - denoiser)
        xnext  = n_ipt - self.gamma * (delta_g + xSubD) # torch.Size([1, 1, H, W, 2])
        # if gt is not None:
        #     batch_pre = torch.unbind(iFFT(xnext), -1)[0].detach()
        #     rsnr = compute_rsnr(gt.squeeze().detach().cpu().numpy(),batch_pre.squeeze().detach().cpu().numpy())[0]
        #     print('rsnr_iters:   ', rsnr)
        return xnext

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, batch_size, emParams, **kwargs):
        super().__init__()
        self.f = f
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.emParams = emParams
        self.index_choose = index_choose_(index_sets=emParams['index_sets'])
        
    def forward(self, n_ipt, n_y=None, gt=None, create_graph=True, strict=True):
        tauList, gammaList = None, None
        n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
        n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
        n_ipt = torch.stack([n_ipt_real, torch.zeros_like(n_ipt_real)], -1)
        n_ipt = FFT(n_ipt)
        n_ipt_periter = n_ipt_real[0,:]
        # compute forward pass and re-engage autograd tape

        # sub =  torch.randperm(self.emParams['NBFkeep']).tolist()
        # meas_list = self.index_choose.get_subset_radial(batch_size=self.batch_size)#sub[0:self.batch_size]
        Hreal_1 = self.emParams['Hreal'][0].shape[0]
        Hreal_2 = self.emParams['Hreal'][1].shape[0]
        
        sub1 =  torch.randperm(Hreal_1)
        sub2 =  torch.randperm(Hreal_2)
        meas_list_1 = sub1[0:Hreal_1]
        meas_list_2 = sub2[0:self.batch_size-Hreal_1] 

        meas_list = torch.cat([meas_list_1, (meas_list_2+Hreal_1)])

        # meas_list = self.index_choose.get_subset_random(NBFkeep=self.emParams['NBFkeep'], batch_size=self.batch_size)#sub[0:self.batch_size]

        emStoc = {}
        emStoc['NBFkeep'] = len(meas_list)
        emStoc['Hreal'] = torch.cat([self.emParams['Hreal'][0][meas_list_1,:].to(n_ipt.device), self.emParams['Hreal'][1][meas_list_2,:].to(n_ipt.device)])
        # emStoc['Himag'] = emParams['Himag'][meas_list,:]
        yStoc = n_y[:,meas_list,...].to(n_ipt.device)

        #with torch.no_grad():
        z, self.forward_res, _ = self.solver_img(lambda z : self.f(z, yStoc, emStoc, meas_list, gt=gt, create_graph=False, strict=False), n_ipt, **self.kwargs)
        z = self.f(z, yStoc, emStoc, meas_list, gt=gt, create_graph=create_graph, strict=strict)
        # set up Jacobian vector product (without additional forward calls)
        if create_graph:
            z0 = z.clone().detach().requires_grad_()
            # f0 = self.f(z0, yStoc, emStoc, meas_list, create_graph=create_graph, strict=strict)
            r0 = self.f.gamma * self.f.tau * self.f.denoise(z0, create_graph=create_graph, strict=strict)
            def backward_hook(grad):
                fTg = lambda y : y - self.f.gamma*self.f.dObj.fwd_bwd(y, emStoc) - autograd.grad(r0, z0, y, retain_graph=True)[0] + grad
                g, self.backward_res = self.solver_grad(fTg, grad, **self.kwargs)
                return g
            z.register_hook(backward_hook)
        z = iFFT(z)
        z[...,-1] = 0
        return z, tauList, gammaList, n_ipt_periter#, loss_fp_dist

class Unfold_model(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""
    def __init__(self, dnn: nn.Module, config):
        super(Unfold_model, self).__init__()
        self.f = dnn
        self.mode = config['unfold_model'].mode
        self.num_iter = config['unfold_model'].num_iter

    def forward(self, n_ipt, n_y=None, emParams=None, fgrad=None):
        tauList, gammaList = [], []
        padding = nn.ReflectionPad2d(1)
        n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
        n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
        n_ipt = torch.stack([n_ipt_real, torch.zeros_like(n_ipt_real)], -1)
        n_ipt = FFT(n_ipt)    
        if self.mode == 'onRED':
            n_ipt_periter = n_ipt_real[0,:]
            # mode == 1 : implement Red training input: H'y target : gt
            for i in range(self.num_iter):
                n_ipt = self.f(n_ipt, n_y, emParams, fgrad)
                tauList.append(self.f.tau)
                gammaList.append(self.f.gamma)
                n_ipt_temp = torch.unbind(iFFT(n_ipt), -1)[0]
                n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)
            n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
            n_ipt_periter = torch.unsqueeze(n_ipt_periter,dim=1)
        elif self.mode=='onPnP':
            # mode == 2 : implement Plug and play training input: H'y target : gt
            for i in range(self.num_iter):
                sub =  torch.randperm(emParams['NBFkeep']).tolist()
                meas_list = sub[0:self.batch_size]
                delta_g = fgrad(n_ipt, n_y, meas_list, emParams)                
                delta_g = fgrad(n_ipt, n_y, emParams)
                z_ipt  = n_ipt - gammaList[i] * (delta_g)
                z_ipt_real = torch.unbind(iFFT(z_ipt), -1)[0]
                z_ipt_real, max_, min_ = tensor_normlize(z_ipt_real)
                denoiser = self.dnn(z_ipt_real)
                denoiser = tensor_denormlize(denoiser, max_, min_)
                denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
                n_ipt = FFT(denoiser)
                tauList.append(tauList[i])
                gammaList.append(gammaList[i])                
            n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
        else:
            raise Exception("Unrecognized mode.")
        return n_ipt, tauList, gammaList, n_ipt_periter

class sgdnetIDT(torch.nn.Module):
    """Train SGD-Net/Bach-Unfold with pixel loss"""
    def __init__(self, config, device):
        super(sgdnetIDT, self).__init__()
        print(OmegaConf.to_yaml(config))
        self.config = config
        self.device = device
        self.opt_train = config['training']
        self.save_path = config['save_path']
        self.set_mode = self.config['set_mode']
        
    def init_state(self, train_set=None, valid_set=None, test_set=None):

        '''
        ----------------------------------------
        (creat dataloader)
        ----------------------------------------
        '''

        if test_set is not None:
            self.emParamsTest = test_set['emParamsTest']
            test_dataset = TestDataset(test_set['test_ipt'], 
                           test_set['test_gdt'], test_set['test_y'])

            self.test_dataLoader = DataLoader(test_dataset, 
                        batch_size=self.config['testing'].batch_size)
            # self.fgrad_SGD_test = test_dataset.fgrad_SGD            
        elif train_set and valid_set is not None:
            self.emParamsTrain = train_set['emParamsTrain']
            # path_ = '/export/project/jiaming.liu/Projects/potential_SDEQ_IDT/data/IDT/imgs/MRI_HW=320_NBFkeep=240_top.h5'
            train_dataset = TrainDataset(train_set['train_ipt'],
                       train_set['train_gdt'], train_set['train_y'])
            # self.fgrad_SGD_train = train_dataset.fgrad_SGD
            self.train_dataLoader = DataLoader(train_dataset, 
            batch_size=self.config['training'].batch_size, shuffle=True)
            self.emParamsValid = valid_set['emParamsValid'] 
            valid_dataset = ValidDataset(valid_set['valid_ipt'], 
                            valid_set['valid_gdt'], valid_set['valid_y'])
            self.valid_dataLoader = DataLoader(valid_dataset, 
                         batch_size=self.config['validating'].batch_size)
            # self.fgrad_SGD_valid = valid_dataset.fgrad_SGD             
        else:
            raise Exception("Unrecognized dataset.")
        '''
        ----------------------------------------
        (initialize model)
        ----------------------------------------
        '''
        
        self.define_model()

        if self.set_mode=='train':  
            self.define_loss()    
            self.define_optimizer()
            self.define_scheduler()
            self.global_step = 0
            self.writer = SummaryWriter(log_dir=self.save_path+"/logs")
            self.save_log()

    # ----------------------------------------
    # define unfold network (SGD-Net)
    # ----------------------------------------
    def define_model(self):
        self.dnn = net_model(self.config['cnn_model'])
        self.dnn.apply(weights_init_kaiming)

        if self.set_mode == 'test':
            fwd_set = self.config['fwd_test']
        else:
            fwd_set = self.config['fwd_train']   
            if self.config['warm_up']:
                self.load_warmup()
        self.URED = URED(self.dnn, self.config['unfold_model'], batch_size=fwd_set['batchSize'])
        self.network =  DEQFixedPoint(self.URED, anderson_img, anderson_grad, 
                batch_size=fwd_set['batchSize'], emParams=self.emParamsTrain, tol=1e-4, max_iter=60).to(self.device)
        # self.network = nn.DataParallel(self.network, device_ids=[0,1]).to(self.device)
        # self.network =  Unfold_model(self.URED, self.config).to(self.device)
        # self.network =  Unfold_model(self.dnn, self.config, batch_size=fwd_set['batchSize'], **self.config['unfold_model']).to(self.device)
        # TODO: design multiGPU for IDT.
        if self.config['multiGPU']:
            gpu_ids = [t for t in range(self.config['num_gpus'])]
            self.network = nn.DataParallel(self.network, device_ids=gpu_ids).to(self.device)
    # ----------------------------------------
    # define loss function
    # ----------------------------------------
    def define_loss(self):
        lossfn_type = self.config['training'].loss
        if lossfn_type == 'l1':
            self.lossfn = nn.L1Loss().to(self.device)
        elif lossfn_type == 'l2':
            self.lossfn = nn.MSELoss().to(self.device)
        elif lossfn_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum').to(self.device)  
        elif lossfn_type == 'ssim':
            self.lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):

        if self.opt_train['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), 
                    lr=self.opt_train['inti_lr'], weight_decay=self.opt_train['weight_decay'])
        elif self.opt_train['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(), 
                                        lr=self.opt_train['inti_lr'], 
                                        momentum=self.opt_train['momentum'],
                                        nesterov=self.opt_train['nesterov'],
                                        weight_decay=self.opt_train['weight_decay'])
        print(self.optimizer)                                
    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_scheduler(self):
        if self.config['training'].optimizer == "Adam":
            self.scheduler = None
        elif self.config['training'].optimizer == "SGD":
            if self.config['training'].scheduler == "Multistep":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,80,120,160,200,250], gamma=0.7)
            elif self.config['training'].scheduler == "CyclicLR":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                        step_size_up=250, base_lr=0.05, max_lr=0.3,mode='exp_range')


    # ----------------------------------------
    # load pretrained for test
    # ----------------------------------------
    def load_test(self):
        load_path = os.path.join(self.config['root_path'], self.config['inference'].load_path)

        try:
            checkpoint = torch.load(load_path)['model_state_dict'] 
        except KeyError:
            checkpoint = torch.load(load_path)
        except Exception as err:
            print('Error occured at', err)
            exit()

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = 'f.'+k # remove `module.`
            new_state_dict[name] = v

        try:
            self.network.load_state_dict(new_state_dict,strict=True)
        except RuntimeError:
            self.network.state_dict()['tau'] = torch.tensor(self.config['unfold_model'].tau_inti, dtype=torch.float32)
            checkpoint = torch.load(load_path)
        except Exception as err:
            print('Error occured at', err)
            exit()

    # ----------------------------------------
    # load pretrained for warmup
    # ----------------------------------------
    def load_warmup(self):
        # load_path = os.path.join(self.config['root_path'], self.config['keep_training'].load_path)
        load_path = os.path.join(self.config['keep_training'].load_path)
        checkpoint = torch.load(load_path)
        
        try:
            self.dnn.load_state_dict(checkpoint['model_state_dict'],strict=True)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.dnn.load_state_dict(new_state_dict,strict=True)
            print("Succsfully load warmup")
        except Exception as err:
            print('Error occured at', err)

    # ----------------------------------------
    # Test the pre-trained models
    # ----------------------------------------
    def test(self):
        
        self.load_test()
        count, sum_snr, sum_rsnr = 0, 0, 0
        self.network.eval()

        with torch.no_grad():

            for batch_ipt, batch_gdt, batch_y in tqdm(self.test_dataLoader):

                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_pre, _, _, batch_pre_iter = self.network(batch_ipt, batch_y, self.emParamsTest, self.fgrad_SGD_test)

                for minibatch in range(batch_pre.shape[0]):
                    snr = compare_snr(batch_pre[minibatch,...], batch_gdt[minibatch,...])
                    rsnr = compute_rsnr(batch_gdt[minibatch,...].squeeze().detach().cpu().numpy(),batch_pre[minibatch,...].squeeze().detach().cpu().numpy())[0]
                    sum_snr = sum_snr + snr.item()
                    sum_rsnr = sum_rsnr + rsnr
                    count = count + 1
        Avg_snr = sum_snr/count     
        Avg_rsnr = sum_rsnr/count  
        print("==================================\n")
        print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
        print('Avg SNR: ', Avg_snr)
        print('Avg rSNR: ', Avg_rsnr)
        print("\n==================================")
        print("")
    # ----------------------------------------
    # save text into tensorboard
    # ----------------------------------------
    def save_log(self):

        temp = OmegaConf.to_yaml(self.config['measure_path'])
        temp = temp.split('\n')
        for count, value in enumerate(temp):
            # print(count, ' ', value.replace(' ', ''))
            self.writer.add_text('measure_path', value.replace(' ', ''), 0)

        self.writer.flush()
    # # --------------------------------------
    # save the state_dict of the network
    # ----------------------------------------     
    def save_model(self, epoch):

        torch.save({
            'global_step':self.global_step,
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.lossfn,
        }, self.save_path+'/logs/unfold_red_epoch%d.pth'%(epoch))
    # ----------------------------------------
    # define loss function
    # ----------------------------------------
    def define_loss(self):
        lossfn_type = self.config['training'].loss
        if lossfn_type == 'l1':
            self.lossfn = nn.L1Loss().to(self.device)
        elif lossfn_type == 'l2':
            self.lossfn = nn.MSELoss().to(self.device)
        elif lossfn_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum').to(self.device)  
        elif lossfn_type == 'ssim':
            self.lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------    
    def train_optimize(self):

        for epoch in range(1, self.opt_train['unfold_epoch']):
            
            loss_count, snr_count = 0,0
            avg_snr_training, avg_rsnr_training, avg_loss_training = 0,0,0

            self.network.train()
            # if epoch >= self.opt_train['unfold_lr_milstone'] and epoch % 15 == 0:
            #     current_lr = self.opt_train['unfold_lr'] *0.7
            #     # adjust learning rate
            #     print('current_lr: ', current_lr,'epoch: ', epoch)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = current_lr
            # set learning rate
            print('learning rate %f' % self.optimizer.param_groups[0]['lr'])
            for batch_ipt, batch_gdt, batch_y in tqdm(self.train_dataLoader):

                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)

                batch_gdt_iFFT = torch.stack([batch_gdt, torch.zeros_like(batch_gdt)], -1).detach()

                self.optimizer.zero_grad()
                batch_pre_iFFT, _, gammaList, batch_pre_iter = self.network(batch_ipt, batch_y, gt=batch_gdt)
                batch_pre = torch.unbind(batch_pre_iFFT, -1)[0].detach()
                loss = self.lossfn(batch_pre_iFFT, batch_gdt_iFFT)
                avg_loss_training = loss.item() + avg_loss_training
                loss.backward()
                
                snr = 0
                for i_test in range(batch_gdt.shape[0]):
                    snr_each =  compare_snr(batch_pre[i_test,...].squeeze(), batch_gdt[i_test,...].squeeze())
                    snr = snr + snr_each
                    avg_snr_training = avg_snr_training + snr_each
                    snr_count = snr_count + 1
                snr = snr.item() / batch_gdt.shape[0]    
                self.writer.add_scalar('train/snr_iter', snr, self.global_step)
                self.writer.add_scalar('train/loss_iter', loss.item(), self.global_step)


                if self.global_step % 10 ==0:
                    grads_min = []
                    grads_max = []        
                    for param in self.optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            grads_min.append(torch.min(param.grad))
                            grads_max.append(torch.max(param.grad))
                    grads_min = torch.min(torch.stack(grads_min,0))
                    grads_max = torch.max(torch.stack(grads_max,0))
                    self.writer.add_scalar('train/grad_iter_max_1', grads_max.item(), self.global_step)
                    self.writer.add_scalar('train/grad_iter_min_1', grads_min.item(), self.global_step)

                if epoch % 3 == 0:
                    for i_test in range(batch_gdt.shape[0]):
                        rsnr = compute_rsnr(batch_gdt.squeeze().detach().cpu().numpy(),batch_pre.squeeze().detach().cpu().numpy())[0]
                        avg_rsnr_training = avg_rsnr_training + rsnr

                torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=2e-4)
                self.optimizer.step()

                # if self.scheduler and  epoch < self.opt_train['unfold_lr_milstone']:
                #     self.scheduler.step()

                loss_count += 1
                self.global_step += 1

            if self.scheduler:
                self.scheduler.step()

            if epoch % 3 == 0:
                avg_rsnr_training = avg_rsnr_training / snr_count
                self.writer.add_scalar('train/rsnr_epoch', avg_rsnr_training, epoch)
            with torch.no_grad():
                avg_snr_training = avg_snr_training / snr_count
                avg_loss_training = avg_loss_training / loss_count 
                batch_pre = torch.clamp(batch_pre[0:2],0,1)
                batch_gdt = torch.clamp(batch_gdt[0:2],0,1)
                batch_ipt_real = torch.unbind(iFFT(batch_ipt[0].unsqueeze(0)), -1)[0]
                batch_ipt_real, max_, min_ = tensor_normlize(batch_ipt_real)
                batch_ipt_real = torch.clamp(batch_ipt_real,0,1)                 
                Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=False, scale_each=True)
                Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=False, scale_each=True)
                Img_ipt = utils.make_grid(batch_ipt_real, nrow=3, normalize=False, scale_each=True)
                # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
                self.writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
                self.writer.add_image('train/ipt', Img_ipt, epoch, dataformats='CHW')
                self.writer.add_scalar('train/snr_epoch', avg_snr_training, epoch)
                self.writer.add_scalar('train/loss_epoch', avg_loss_training, epoch)
                self.writer.add_histogram('train/histogrsm', batch_pre, epoch)
                # self.writer.add_histogram('train/gammaList', np.asarray(gammaList), epoch)    
            # #####################################
            # #               Valid               # 
            # #####################################
            self.network.eval()
            snr_avg = 0
            rsnr_avg = 0
            count = 0
            for batch_ipt, batch_gdt, batch_y in self.valid_dataLoader:
                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)
                batch_pre_iFFT, _, _, batch_pre_iter = self.network(batch_ipt, batch_y, create_graph=False, strict=False)
                batch_pre = torch.unbind(batch_pre_iFFT, -1)[0].detach()
                for i_test in range(batch_gdt.shape[0]):
                    snr_avg =  snr_avg + compare_snr(batch_pre[i_test,...].squeeze(), batch_gdt[i_test,...].squeeze())
                    rsnr_avg = rsnr_avg + compute_rsnr(batch_gdt[i_test,...].squeeze().detach().cpu().numpy(),batch_pre[i_test,...].squeeze().detach().cpu().numpy())[0]
                    count = count + 1
            with torch.no_grad():            
                snr_avg = snr_avg / count
                rsnr_avg = rsnr_avg / count
                self.writer.add_scalar('valid/rsnr',rsnr_avg.item(), epoch)
                self.writer.add_scalar('valid/snr',snr_avg.item(), epoch)
                self.writer.add_histogram('valide/histogrsm', batch_pre, epoch)
                print("==================================\n")
                print("epoch: [%d]" % epoch)
                print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
                print('snr: ', snr_avg.item(), 'rsnr: ', rsnr_avg)
                print("\n==================================")
                print("")                
                batch_pre = torch.clamp(batch_pre[0:2],0,1)
                batch_gdt = torch.clamp(batch_gdt[0:2],0,1)        
                # batch_pre_iter = torch.clamp(batch_pre_iter,0,1)             
                Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=False, scale_each=True)
                Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=False, scale_each=True)
                # Img_ipt = utils.make_grid(batch_ipt, nrow=4, normalize=False, scale_each=True)
                # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
                # self.writer.add_image('valid/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
                self.writer.add_image('valid/pre', Img_pre, epoch, dataformats='CHW')
                self.writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')
                # writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')

            if epoch % self.opt_train['save_epoch'] == 0:
                self.save_model(epoch)
        self.writer.close()
        ###############################################
        #                  End Training               #
        ###############################################

# from omegaconf import DictConfig, OmegaConf
# from collections import OrderedDict

# import torch
# from torch import nn
# from torch import optim
# import torchvision.utils as utils
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

# import numpy as np
# from tqdm import tqdm

# from models.loss_ssim import SSIMLoss
# from utils.net_mode import *
# from utils.util import *

# from DataFidelities.Dataclass import *

# from DataFidelities.IDTClass import *
# import torch.autograd as autograd

# ################Functions######################
# FFT  = lambda x: torch.fft(x,  signal_ndim=2)
# iFFT = lambda x: torch.ifft(x, signal_ndim=2)
# rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
# ###############################################

# def tensor_normlize(n_ipt:torch.tensor):
#     if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
#         dim_x, dim_y = n_ipt.shape[2:4]
#         n_ipt_max = torch.max(n_ipt.max(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
#         n_ipt_min = torch.min(n_ipt.min(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
#         n_ipt_norm = (n_ipt - n_ipt_min) / (n_ipt_max - n_ipt_min)
#     return n_ipt_norm, n_ipt_max, n_ipt_min

# def tensor_denormlize(n_ipt, n_ipt_max, n_ipt_min):
#     if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
#         dim_x, dim_y = n_ipt.shape[2:4]
#         n_ipt_denorm = torch.mul(n_ipt, n_ipt_max-n_ipt_min) + n_ipt_min
#     return n_ipt_denorm

# def anderson_img(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1):
#     """ Anderson acceleration for fixed point iteration. """
#     bsz, d, H, W, CX = x0.shape
#     X = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
#     F = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
#     X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
#     X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
#     H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
#     H[:,0,1:] = H[:,1:,0] = 1
#     y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
#     y[:,0] = 1
    
#     res = []
#     for k in range(2, max_iter):
#         n = min(k, m)
#         G = F[:,:n]-X[:,:n]
#         H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
#         alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
#         X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
#         F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
#         res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
#         if (res[-1] < tol):
#             break

#         # if k%10==0:
#         #     n_ipt_temp = torch.unbind(iFFT(X[:,k%m].view_as(x0)), -1)[0]
#         #     n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)

#     return X[:,k%m].view_as(x0), res, None#n_ipt_periter

# def anderson_grad(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1.0):
#     """ Anderson acceleration for fixed point iteration. """
#     bsz, d, H, W, CX = x0.shape
#     X = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
#     F = torch.zeros(bsz, m, d*H*W*CX, dtype=x0.dtype, device=x0.device)
#     X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
#     X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
#     H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
#     H[:,0,1:] = H[:,1:,0] = 1
#     y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
#     y[:,0] = 1
    
#     res = []
#     for k in range(2, max_iter):
#         n = min(k, m)
#         G = F[:,:n]-X[:,:n]
#         H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
#         alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
#         X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
#         F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
#         res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
#         if (res[-1] < tol):
#             break

#     return X[:,k%m].view_as(x0), res

# class URED(nn.Module):

#     def __init__(self, dnn: nn.Module, config:dict, batch_size=40):
#         #gamma_inti=3e-3, tau_inti=1e-1, batch_size=60, device='cuda:0'):

#         super(URED, self).__init__()
#         self.dnn = dnn

#         self.gamma = torch.tensor(config['gamma_inti'], dtype=torch.float32)
#         self.tau = torch.tensor(config['tau_inti'], dtype=torch.float32) #torch.nn.Parameter(torch.tensor(config['tau_inti'], dtype=torch.float32, requires_grad=True))
#         self.dObj = IDTClass()
        
#     def forward(self, n_ipt, n_y=None, emParams=None, meas_list=None, gt=None):
#         delta_g = self.dObj.fgrad_SGD(n_ipt, n_y, meas_list, emParams)
#         n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
#         n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
#         denoiser = self.dnn(n_ipt_real)        
#         denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
#         denoiser = FFT(denoiser)
#         xSubD    = self.tau * (n_ipt - denoiser)
#         xnext  = n_ipt - self.gamma * (delta_g + xSubD) # torch.Size([1, 1, H, W, 2])
#         # psnr_iters = compare_psnr(xnext.squeeze().detach().clone(), gt.squeeze().detach().clone()).item()
#         # print('psnr_iters:   ', psnr_iters)
#         return xnext

# class DEQFixedPoint(nn.Module):
#     def __init__(self, f, solver_img, solver_grad, batch_size, emParams, **kwargs):
#         super().__init__()
#         self.f = f
#         self.solver_img = solver_img
#         self.solver_grad = solver_grad
#         self.batch_size = batch_size
#         self.kwargs = kwargs
#         self.emParams = emParams
        
#     def forward(self, n_ipt, n_y=None):
#         tauList, gammaList = None, None
#         n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
#         n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
#         n_ipt = torch.stack([n_ipt_real, torch.zeros_like(n_ipt_real)], -1)
#         n_ipt = FFT(n_ipt)
#         n_ipt_periter = n_ipt_real[0,:]
#         # compute forward pass and re-engage autograd tape

#         sub =  torch.randperm(self.emParams['NBFkeep']).tolist()
#         meas_list = sub[0:self.batch_size]

#         emStoc = {}
#         emStoc['NBFkeep'] = self.batch_size
#         emStoc['Hreal'] = self.emParams['Hreal'][meas_list,:].to(n_ipt.device)  
#         # emStoc['Himag'] = emParams['Himag'][meas_list,:]
#         yStoc = n_y[:,meas_list,...].to(n_ipt.device)

#         with torch.no_grad():
#             z, self.forward_res, _ = self.solver_img(lambda z : self.f(z, yStoc, emStoc, meas_list), n_ipt, **self.kwargs)
        
#         z = self.f(z, yStoc, emStoc, meas_list)
#         # set up Jacobian vector product (without additional forward calls)
#         z0 = z.clone().detach().requires_grad_()
#         f0 = self.f(z0, yStoc, emStoc, meas_list)
#         def backward_hook(grad):
#             g, self.backward_res = self.solver_grad(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
#                                                grad, **self.kwargs)
#             return g

#         z.register_hook(backward_hook)

#         # z = iFFT(z)
#         # z[...,-1] = 0

#         return z, tauList, gammaList, n_ipt_periter#, loss_fp_dist

# class Unfold_model(nn.Module):
#     """Unfold network models, i.e. (online) PnP/RED"""
#     def __init__(self, dnn: nn.Module, config):
#         super(Unfold_model, self).__init__()
#         self.f = dnn
#         self.mode = config['unfold_model'].mode
#         self.num_iter = config['unfold_model'].num_iter

#     def forward(self, n_ipt, n_y=None, emParams=None, fgrad=None):
#         tauList, gammaList = [], []
#         padding = nn.ReflectionPad2d(1)
#         n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
#         n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
#         n_ipt = torch.stack([n_ipt_real, torch.zeros_like(n_ipt_real)], -1)
#         n_ipt = FFT(n_ipt)    
#         if self.mode == 'onRED':
#             n_ipt_periter = n_ipt_real[0,:]
#             # mode == 1 : implement Red training input: H'y target : gt
#             for i in range(self.num_iter):
#                 n_ipt = self.f(n_ipt, n_y, emParams, fgrad)
#                 tauList.append(self.f.tau)
#                 gammaList.append(self.f.gamma)
#                 n_ipt_temp = torch.unbind(iFFT(n_ipt), -1)[0]
#                 n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)
#             n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
#             n_ipt_periter = torch.unsqueeze(n_ipt_periter,dim=1)
#         elif self.mode=='onPnP':
#             # mode == 2 : implement Plug and play training input: H'y target : gt
#             for i in range(self.num_iter):
#                 sub =  torch.randperm(emParams['NBFkeep']).tolist()
#                 meas_list = sub[0:self.batch_size]
#                 delta_g = fgrad(n_ipt, n_y, meas_list, emParams)                
#                 delta_g = fgrad(n_ipt, n_y, emParams)
#                 z_ipt  = n_ipt - gammaList[i] * (delta_g)
#                 z_ipt_real = torch.unbind(iFFT(z_ipt), -1)[0]
#                 z_ipt_real, max_, min_ = tensor_normlize(z_ipt_real)
#                 denoiser = self.dnn(z_ipt_real)
#                 denoiser = tensor_denormlize(denoiser, max_, min_)
#                 denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
#                 n_ipt = FFT(denoiser)
#                 tauList.append(tauList[i])
#                 gammaList.append(gammaList[i])                
#             n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
#         else:
#             raise Exception("Unrecognized mode.")
#         return n_ipt, tauList, gammaList, n_ipt_periter

# class sgdnetIDT(torch.nn.Module):
#     """Train SGD-Net/Bach-Unfold with pixel loss"""
#     def __init__(self, config, device):
#         super(sgdnetIDT, self).__init__()
#         print(OmegaConf.to_yaml(config))
#         self.config = config
#         self.device = device
#         self.opt_train = config['training']
#         self.save_path = config['save_path']
#         self.set_mode = self.config['set_mode']
        
#     def init_state(self, train_set=None, valid_set=None, test_set=None):

#         '''
#         ----------------------------------------
#         (creat dataloader)
#         ----------------------------------------
#         '''

#         if test_set is not None:
#             self.emParamsTest = test_set['emParamsTest']
#             test_dataset = TestDataset(test_set['test_ipt'], 
#                            test_set['test_gdt'], test_set['test_y'])

#             self.test_dataLoader = DataLoader(test_dataset, 
#                         batch_size=self.config['testing'].batch_size)
#             # self.fgrad_SGD_test = test_dataset.fgrad_SGD            
#         elif train_set and valid_set is not None:
#             self.emParamsTrain = train_set['emParamsTrain']
#             train_dataset = TrainDataset(train_set['train_ipt'], train_set['train_gdt'], train_set['train_y'])
#             # self.fgrad_SGD_train = train_dataset.fgrad_SGD
#             self.train_dataLoader = DataLoader(train_dataset, 
#             batch_size=self.config['training'].batch_size, shuffle=True)
#             self.emParamsValid = valid_set['emParamsValid'] 
#             valid_dataset = ValidDataset(valid_set['valid_ipt'], 
#                             valid_set['valid_gdt'], valid_set['valid_y'])
#             self.valid_dataLoader = DataLoader(valid_dataset, 
#                          batch_size=self.config['validating'].batch_size)
#             # self.fgrad_SGD_valid = valid_dataset.fgrad_SGD             
#         else:
#             raise Exception("Unrecognized dataset.")
#         '''
#         ----------------------------------------
#         (initialize model)
#         ----------------------------------------
#         '''
        
#         self.define_model()

#         if self.set_mode=='train':  
#             self.define_loss()    
#             self.define_optimizer()
#             self.define_scheduler()
#             self.global_step = 0
#             self.writer = SummaryWriter(log_dir=self.save_path+"/logs")

#     # ----------------------------------------
#     # define unfold network (SGD-Net)
#     # ----------------------------------------
#     def define_model(self):
#         self.dnn = net_model(self.config['cnn_model'])
#         self.dnn.apply(weights_init_kaiming)

#         if self.set_mode == 'test':
#             fwd_set = self.config['fwd_test']
#         else:
#             fwd_set = self.config['fwd_train']   
#             if self.config['warm_up']:
#                 self.load_warmup()
#         self.URED = URED(self.dnn, self.config['unfold_model'], batch_size=fwd_set['batchSize']).to(device)
#         self.network =  DEQFixedPoint(self.URED, anderson_img, anderson_grad, 
#                 batch_size=fwd_set['batchSize'], emParams=self.emParamsTrain, tol=1e-4, max_iter=60).to(self.device)
#         self.network = nn.DataParallel(self.network, device_ids=[0,1]).to(self.device)
#         # self.network =  Unfold_model(self.URED, self.config).to(self.device)
#         # self.network =  Unfold_model(self.dnn, self.config, batch_size=fwd_set['batchSize'], **self.config['unfold_model']).to(self.device)
#         # TODO: design multiGPU for IDT.
#         # if self.config['multiGPU']:
#         #     gpu_ids = [t for t in range(self.config['num_gpus'])]
#         #     self.network = nn.DataParallel(self.network, device_ids=gpu_ids).to(self.device)
#     # ----------------------------------------
#     # define loss function
#     # ----------------------------------------
#     def define_loss(self):
#         lossfn_type = self.config['training'].loss
#         if lossfn_type == 'l1':
#             self.lossfn = nn.L1Loss().to(self.device)
#         elif lossfn_type == 'l2':
#             self.lossfn = nn.MSELoss().to(self.device)
#         elif lossfn_type == 'l2sum':
#             self.lossfn = nn.MSELoss(reduction='sum').to(self.device)  
#         elif lossfn_type == 'ssim':
#             self.lossfn = SSIMLoss().to(self.device)
#         else:
#             raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
#     # ----------------------------------------
#     # define optimizer
#     # ----------------------------------------
#     def define_optimizer(self):

#         if self.opt_train['optimizer'] == 'Adam':
#             self.optimizer = optim.Adam(self.network.parameters(), 
#                     lr=self.opt_train['inti_lr'], weight_decay=self.opt_train['weight_decay'])
#         elif self.opt_train['optimizer'] == 'SGD':
#             self.optimizer = optim.SGD(self.network.parameters(), 
#                                         lr=self.opt_train['inti_lr'], 
#                                         momentum=self.opt_train['momentum'],
#                                         nesterov=self.opt_train['nesterov'],
#                                         weight_decay=self.opt_train['weight_decay'])
#     # ----------------------------------------
#     # define optimizer
#     # ----------------------------------------
#     def define_scheduler(self):
#         if self.config['training'].optimizer == "Adam":
#             pass
#         elif self.config['training'].optimizer == "SGD":
#             self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
#                             base_lr=0.004, max_lr=0.008,mode='exp_range')

#     # ----------------------------------------
#     # load pretrained for test
#     # ----------------------------------------
#     def load_test(self):
#         load_path = os.path.join(self.config['root_path'], self.config['inference'].load_path)

#         try:
#             checkpoint = torch.load(load_path)['model_state_dict'] 
#         except KeyError:
#             checkpoint = torch.load(load_path)
#         except Exception as err:
#             print('Error occured at', err)
#             exit()

#         new_state_dict = OrderedDict()
#         for k, v in checkpoint.items():
#             name = 'f.'+k # remove `module.`
#             new_state_dict[name] = v

#         try:
#             self.network.load_state_dict(new_state_dict,strict=True)
#         except RuntimeError:
#             self.network.state_dict()['tau'] = torch.tensor(self.config['unfold_model'].tau_inti, dtype=torch.float32)
#             checkpoint = torch.load(load_path)
#         except Exception as err:
#             print('Error occured at', err)
#             exit()

#     # ----------------------------------------
#     # load pretrained for warmup
#     # ----------------------------------------
#     def load_warmup(self):
#         load_path = os.path.join(self.config['root_path'], self.config['keep_training'].load_path)
#         checkpoint = torch.load(load_path)
        

#         try:
#             self.dnn.load_state_dict(checkpoint['model_state_dict'],strict=True)
#         except RuntimeError:
#             new_state_dict = OrderedDict()
#             for k, v in checkpoint['model_state_dict'].items():
#                 name = k[7:] # remove `module.`
#                 new_state_dict[name] = v
#             self.dnn.load_state_dict(new_state_dict,strict=True)
#             print("Succsfully load warmup")
#         except Exception as err:
#             print('Error occured at', err)

#     # ----------------------------------------
#     # Test the pre-trained models
#     # ----------------------------------------
#     def test(self):
        
#         self.load_test()
#         count, sum_snr, sum_rsnr = 0, 0, 0
#         self.network.eval()

#         with torch.no_grad():

#             for batch_ipt, batch_gdt, batch_y in tqdm(self.test_dataLoader):

#                 batch_ipt = batch_ipt.to(self.device)
#                 batch_gdt = batch_gdt.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 batch_pre, _, _, batch_pre_iter = self.network(batch_ipt, batch_y, self.emParamsTest, self.fgrad_SGD_test)

#                 for minibatch in range(batch_pre.shape[0]):
#                     snr = compare_snr(batch_pre[minibatch,...], batch_gdt[minibatch,...])
#                     rsnr = compute_rsnr(batch_gdt[minibatch,...].squeeze().detach().cpu().numpy(),batch_pre[minibatch,...].squeeze().detach().cpu().numpy())[0]
#                     sum_snr = sum_snr + snr.item()
#                     sum_rsnr = sum_rsnr + rsnr
#                     count = count + 1
#         Avg_snr = sum_snr/count     
#         Avg_rsnr = sum_rsnr/count  
#         print("==================================\n")
#         print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
#         print('Avg SNR: ', Avg_snr)
#         print('Avg rSNR: ', Avg_rsnr)
#         print("\n==================================")
#         print("")

#     # # ----------------------------------------
#     # save the state_dict of the network
#     # ----------------------------------------     
#     def save_model(self, epoch):

#         torch.save({
#             'global_step':self.global_step,
#             'epoch': epoch,
#             'model_state_dict': self.network.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': self.lossfn,
#         }, self.save_path+'/logs/unfold_red_epoch%d.pth'%(epoch))

#     # ----------------------------------------
#     # update parameters and get loss
#     # ----------------------------------------    
#     def train_optimize(self):

#         for epoch in range(1, self.opt_train['unfold_epoch']):
#             self.network.train()
#             batch = 0
#             avg_snr_training = 0
#             avg_rsnr_training = 0
#             avg_loss_training = 0
#             if epoch >= self.opt_train['unfold_lr_milstone'] and epoch % 50 == 0:
#                 current_lr = self.optimizer.param_groups[0]['lr'] *0.5
#                 # adjust learning rate
#                 print('current_lr: ', current_lr,'epoch: ', epoch)
#                 for param_group in self.optimizer.param_groups:
#                     param_group['lr'] = current_lr
#             # set learning rate
#             print('learning rate %f' % self.optimizer.param_groups[0]['lr'])
#             for batch_ipt, batch_gdt, batch_y in tqdm(self.train_dataLoader):

#                 batch_ipt = batch_ipt.to(self.device)
#                 batch_gdt = batch_gdt.to(self.device)

#                 batch_gdt_FFT = FFT(torch.stack([batch_gdt, torch.zeros_like(batch_gdt)], -1)).detach()
#                 #torch.stack([batch_gdt, torch.zeros_like(batch_gdt)], -1).detach()#
#                 #FFT(torch.stack([batch_gdt, torch.zeros_like(batch_gdt)], -1)).detach()

#                 self.optimizer.zero_grad()
#                 batch_pre_FFT, tauList, gammaList, batch_pre_iter = self.network(batch_ipt, batch_y)

#                 # print(batch_pre_FFT.shape, batch_gdt_FFT.shape)

#                 loss = self.lossfn(batch_pre_FFT, batch_gdt_FFT)
#                 batch_pre = torch.unbind(iFFT(batch_pre_FFT), -1)[0].detach()
#                 snr  = compare_snr(batch_pre, batch_gdt)
#                 avg_snr_training = snr.item() + avg_snr_training
#                 avg_loss_training = loss.item() + avg_loss_training
#                 self.writer.add_scalar('train/snr', snr.item(), self.global_step)
#                 self.writer.add_scalar('train/loss', loss.item(), self.global_step)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=0.01)
#                 self.optimizer.step()

#                 # if epoch < self.opt_train['unfold_lr_milstone']:
#                 #     self.scheduler.step()

#                 if epoch % 10 == 0:
#                     rsnr = compute_rsnr(batch_gdt.squeeze().detach().cpu().numpy(),batch_pre.squeeze().detach().cpu().numpy())[0]
#                     avg_rsnr_training = avg_rsnr_training + rsnr
#                     # writer.add_scalar('train/rsnr', rsnr, global_step)
            
#                 batch += 1
#                 self.global_step += 1 

#             if epoch % 10 == 0:
#                 avg_rsnr_training = avg_rsnr_training / batch
#                 self.writer.add_scalar('train/rsnr_epoch', avg_rsnr_training, epoch)
#             with torch.no_grad():
#                 avg_snr_training = avg_snr_training / batch
#                 avg_loss_training = avg_loss_training / batch 
#                 batch_pre = torch.clamp(batch_pre,0,1)
#                 batch_gdt = torch.clamp(batch_gdt,0,1)
#                 # batch_pre_iter = torch.clamp(batch_pre_iter,0,1)                 
#                 Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=False, scale_each=True)
#                 Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=False, scale_each=True)
#                 # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
#                 self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
#                 self.writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
#                 self.writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
#                 # self.writer.add_image('train/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
#                 self.writer.add_scalar('train/snr_epoch', avg_snr_training, epoch)
#                 self.writer.add_scalar('train/loss_epoch', avg_loss_training, epoch)
#                 self.writer.add_histogram('train/histogrsm', batch_pre, epoch)
#                 # self.writer.add_histogram('train/tauList', np.asarray(tauList), epoch)
#                 # self.writer.add_histogram('train/gammaList', np.asarray(gammaList), epoch)    
#             # #####################################
#             # #               Valid               # 
#             # #####################################
#             self.network.eval()
#             snr_avg = 0
#             rsnr_avg = 0
#             count = 0
#             for batch_ipt, batch_gdt, batch_y in self.valid_dataLoader:
#                 batch_ipt = batch_ipt.to(self.device)
#                 batch_gdt = batch_gdt.to(self.device)
#                 batch_pre_FFT, _, _, batch_pre_iter = self.network(batch_ipt, batch_y)
#                 batch_pre = batch_pre_FFT[..., 0]#torch.unbind(iFFT(batch_pre_FFT), -1)[0].detach()
#                 for i_test in range(batch_gdt.shape[0]):
#                     snr_avg =  snr_avg + compare_snr(batch_pre[i_test,...].squeeze(), batch_gdt[i_test,...].squeeze())
#                     rsnr_avg = rsnr_avg + compute_rsnr(batch_gdt[i_test,...].squeeze().detach().cpu().numpy(),batch_pre[i_test,...].squeeze().detach().cpu().numpy())[0]
#                     count = count + 1
#             with torch.no_grad():            
#                 snr_avg = snr_avg / count
#                 rsnr_avg = rsnr_avg / count
#                 self.writer.add_scalar('valid/rsnr',rsnr_avg.item(), epoch)
#                 self.writer.add_scalar('valid/snr',snr_avg.item(), epoch)
#                 self.writer.add_histogram('valide/histogrsm', batch_pre, epoch)
#                 print("==================================\n")
#                 print("epoch: [%d]" % epoch)
#                 print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
#                 print('snr: ', snr_avg.item(), 'rsnr: ', rsnr_avg)
#                 print("\n==================================")
#                 print("")                
#                 batch_pre = torch.clamp(batch_pre,0,1)
#                 batch_gdt = torch.clamp(batch_gdt,0,1)        
#                 # batch_pre_iter = torch.clamp(batch_pre_iter,0,1)             
#                 Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=False, scale_each=True)
#                 Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=False, scale_each=True)
#                 # Img_ipt = utils.make_grid(batch_ipt, nrow=4, normalize=False, scale_each=True)
#                 # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
#                 # self.writer.add_image('valid/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
#                 self.writer.add_image('valid/pre', Img_pre, epoch, dataformats='CHW')
#                 self.writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')
#                 # writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')

#             if epoch % self.opt_train['save_epoch'] == 0:
#                 self.save_model(epoch)
#         self.writer.close()
#         ###############################################
#         #                  End Training               #
#         ###############################################

#     # # # ----------------------------------------
#     # # save the state_dict of the network
#     # # ----------------------------------------     
#     # def save_model(self, epoch):

#     #     torch.save({
#     #         'global_step':self.global_step,
#     #         'epoch': epoch,
#     #         'model_state_dict': self.network.state_dict(),
#     #         'optimizer_state_dict': self.optimizer.state_dict(),
#     #         'loss': self.lossfn,
#     #     }, self.save_path+'/logs/unfold_red_epoch%d.pth'%(epoch))
#     # # ----------------------------------------
#     # # Test the pre-trained models
#     # # ----------------------------------------
#     # def test(self):
        
#     #     self.load_test()
#     #     count, sum_snr, sum_rsnr = 0, 0, 0
#     #     self.network.eval()

#     #     with torch.no_grad():

#     #         for batch_ipt, batch_gdt, batch_y in tqdm(self.test_dataLoader):

#     #             batch_ipt = batch_ipt.to(self.device)
#     #             batch_gdt = batch_gdt.to(self.device)
#     #             batch_y = batch_y.to(self.device)
#     #             batch_pre, batch_pre_iter = self.network(batch_ipt, batch_y, self.emParamsTest, self.fgrad_SGD_test)

#     #             for minibatch in range(batch_pre.shape[0]):
#     #                 snr = compare_snr(batch_pre[minibatch,...], batch_gdt[minibatch,...])
#     #                 rsnr = compute_rsnr(batch_gdt[minibatch,...].squeeze().detach().cpu().numpy(),batch_pre[minibatch,...].squeeze().detach().cpu().numpy())[0]
#     #                 sum_snr = sum_snr + snr.item()
#     #                 sum_rsnr = sum_rsnr + rsnr
#     #                 count = count + 1
#     #     Avg_snr = sum_snr/count     
#     #     Avg_rsnr = sum_rsnr/count  
#     #     print("==================================\n")
#     #     print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
#     #     print('Avg SNR: ', Avg_snr)
#     #     print('Avg rSNR: ', Avg_rsnr)
#     #     print("\n==================================")
#     #     print("")
#     # # ----------------------------------------
#     # # update parameters and get loss
#     # # ----------------------------------------    
#     # def train_optimize(self):

#     #     for epoch in range(1, self.opt_train['unfold_epoch']):
#     #         self.network.train()
#     #         batch = 0
#     #         avg_snr_training = 0
#     #         avg_rsnr_training = 0
#     #         avg_loss_training = 0
#     #         if epoch >= self.opt_train['unfold_lr_milstone'] and epoch % 50 == 0:
#     #             current_lr = self.optimizer.param_groups[0]['lr'] *0.5
#     #             # adjust learning rate
#     #             print('current_lr: ', current_lr,'epoch: ', epoch)
#     #             for param_group in self.optimizer.param_groups:
#     #                 param_group['lr'] = current_lr
#     #         # set learning rate
#     #         print('learning rate %f' % self.optimizer.param_groups[0]['lr'])
#     #         for batch_ipt, batch_gdt, batch_y in tqdm(self.train_dataLoader):
#     #             batch_ipt = batch_ipt.to(self.device)
#     #             batch_gdt = batch_gdt.to(self.device)
#     #             self.optimizer.zero_grad()
#     #             batch_pre, batch_pre_iter = self.network(batch_ipt, batch_y, self.emParamsTrain, self.fgrad_SGD_train)
#     #             loss = self.lossfn(batch_pre, batch_gdt)
#     #             snr  = compare_snr(batch_pre, batch_gdt)
#     #             avg_snr_training = snr + avg_snr_training
#     #             avg_loss_training = loss.item() + avg_loss_training
#     #             self.writer.add_scalar('train/snr', snr.item(), self.global_step)
#     #             self.writer.add_scalar('train/loss', loss.item(), self.global_step)
#     #             loss.backward()
#     #             torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=0.01)
#     #             self.optimizer.step()

#     #             if epoch < self.opt_train['unfold_lr_milstone']:
#     #                 self.scheduler.step()

#     #             if epoch % 10 == 0:
#     #                 rsnr = compute_rsnr(batch_gdt.squeeze().detach().cpu().numpy(),batch_pre.squeeze().detach().cpu().numpy())[0]
#     #                 avg_rsnr_training = avg_rsnr_training + rsnr
#     #                 # writer.add_scalar('train/rsnr', rsnr, global_step)
            
#     #             batch += 1
#     #             self.global_step += 1 

#     #         if epoch % 10 == 0:
#     #             avg_rsnr_training = avg_rsnr_training / batch
#     #             self.writer.add_scalar('train/rsnr_epoch', avg_rsnr_training, epoch)
#     #         with torch.no_grad():
#     #             avg_snr_training = avg_snr_training / batch
#     #             avg_loss_training = avg_loss_training / batch 
#     #             batch_pre = torch.clamp(batch_pre,0,1)
#     #             batch_gdt = torch.clamp(batch_gdt,0,1)
#     #             batch_pre_iter = torch.clamp(batch_pre_iter,0,1)                 
#     #             Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=False, scale_each=True)
#     #             Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=False, scale_each=True)
#     #             Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
#     #             self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
#     #             self.writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
#     #             self.writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
#     #             self.writer.add_image('train/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
#     #             self.writer.add_scalar('train/snr_epoch', avg_snr_training.item(), epoch)
#     #             self.writer.add_scalar('train/loss_epoch', avg_loss_training, epoch)
#     #             self.writer.add_histogram('train/histogrsm', batch_pre, epoch)
#     #             # self.writer.add_histogram('train/tauList', np.asarray(tauList), epoch)
#     #             # self.writer.add_histogram('train/gammaList', np.asarray(gammaList), epoch)    
#     #         # #####################################
#     #         # #               Valid               # 
#     #         # #####################################
#     #         self.network.eval()
#     #         snr_avg = 0
#     #         rsnr_avg = 0
#     #         count = 0
#     #         with torch.no_grad():
#     #             for batch_ipt, batch_gdt, batch_y in self.valid_dataLoader:
#     #                 batch_ipt = batch_ipt.to(self.device)
#     #                 batch_gdt = batch_gdt.to(self.device)
#     #                 batch_pre, batch_pre_iter = self.network(batch_ipt, batch_y, self.emParamsValid, self.fgrad_SGD_valid)
#     #                 for i_test in range(batch_gdt.shape[0]):
#     #                     snr_avg =  snr_avg + compare_snr(batch_pre[i_test,...].squeeze(), batch_gdt[i_test,...].squeeze())
#     #                     rsnr_avg = rsnr_avg + compute_rsnr(batch_gdt[i_test,...].squeeze().detach().cpu().numpy(),batch_pre[i_test,...].squeeze().detach().cpu().numpy())[0]
#     #                     count = count + 1
#     #             snr_avg = snr_avg / count
#     #             rsnr_avg = rsnr_avg / count
#     #             self.writer.add_scalar('valid/rsnr',rsnr_avg.item(), epoch)
#     #             self.writer.add_scalar('valid/snr',snr_avg.item(), epoch)
#     #             self.writer.add_histogram('valide/histogrsm', batch_pre, epoch)
#     #             print("==================================\n")
#     #             print("epoch: [%d]" % epoch)
#     #             print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
#     #             print('snr: ', snr_avg.item(), 'rsnr: ', rsnr_avg)
#     #             print("\n==================================")
#     #             print("")                
#     #             batch_pre = torch.clamp(batch_pre,0,1)
#     #             batch_gdt = torch.clamp(batch_gdt,0,1)        
#     #             batch_pre_iter = torch.clamp(batch_pre_iter,0,1)             
#     #             Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=False, scale_each=True)
#     #             Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=False, scale_each=True)
#     #             # Img_ipt = utils.make_grid(batch_ipt, nrow=4, normalize=False, scale_each=True)
#     #             Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=False, scale_each=True)
#     #             self.writer.add_image('valid/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
#     #             self.writer.add_image('valid/pre', Img_pre, epoch, dataformats='CHW')
#     #             self.writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')
#     #             # writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')

#     #         if epoch % self.opt_train['save_epoch'] == 0:
#     #             self.save_model(epoch)
#     #     self.writer.close()
#     #     ###############################################
#     #     #                  End Training               #
#     #     ###############################################    

# # class Unfold_model(nn.Module):
# #     """Unfold network models, i.e. (online) PnP/RED"""
# #     def __init__(self, dnn: nn.Module, config):
# #         super(Unfold_model, self).__init__()
# #         self.f = dnn
# #         self.mode = config['unfold_model'].mode
# #         self.num_iter = config['unfold_model'].num_iter

# #     def forward(self, n_ipt, n_y=None, emParams=None, fgrad=None):
# #         tauList, gammaList = [], []
# #         padding = nn.ReflectionPad2d(1)
# #         n_ipt_real = torch.unbind(iFFT(n_ipt), -1)[0]
# #         n_ipt_real, max_, min_ = tensor_normlize(n_ipt_real)
# #         n_ipt = torch.stack([n_ipt_real, torch.zeros_like(n_ipt_real)], -1)
# #         n_ipt = FFT(n_ipt)    
# #         if self.mode == 'onRED':
# #             n_ipt_periter = n_ipt_real[0,:]
# #             # mode == 1 : implement Red training input: H'y target : gt
# #             for i in range(self.num_iter):
# #                 n_ipt = self.f(n_ipt, n_y, emParams, fgrad)
# #                 tauList.append(self.f.tau)
# #                 gammaList.append(self.f.gamma)
# #                 n_ipt_temp = torch.unbind(iFFT(n_ipt), -1)[0]
# #                 n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)
# #             n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
# #             n_ipt_periter = torch.unsqueeze(n_ipt_periter,dim=1)
# #         elif self.mode=='onPnP':
# #             # mode == 2 : implement Plug and play training input: H'y target : gt
# #             for i in range(self.num_iter):
# #                 sub =  torch.randperm(emParams['NBFkeep']).tolist()
# #                 meas_list = sub[0:self.batch_size]
# #                 delta_g = fgrad(n_ipt, n_y, meas_list, emParams)                
# #                 delta_g = fgrad(n_ipt, n_y, emParams)
# #                 z_ipt  = n_ipt - gammaList[i] * (delta_g)
# #                 z_ipt_real = torch.unbind(iFFT(z_ipt), -1)[0]
# #                 z_ipt_real, max_, min_ = tensor_normlize(z_ipt_real)
# #                 denoiser = self.dnn(z_ipt_real)
# #                 denoiser = tensor_denormlize(denoiser, max_, min_)
# #                 denoiser = torch.stack([denoiser, torch.zeros_like(denoiser)], -1)
# #                 n_ipt = FFT(denoiser)
# #                 tauList.append(tauList[i])
# #                 gammaList.append(gammaList[i])                
# #             n_ipt = torch.unbind(iFFT(n_ipt), -1)[0]
# #         else:
# #             raise Exception("Unrecognized mode.")
# #         return n_ipt, tauList, gammaList, n_ipt_periter        