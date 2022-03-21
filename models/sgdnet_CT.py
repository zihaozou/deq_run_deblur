from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.autograd as autograd

import numpy as np
from tqdm import tqdm

from models.loss_ssim import SSIMLoss
from utils.net_mode import *
from utils.util import *

from DataFidelities.CTClass import *
################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def nesterov(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1):
    """ nesterov acceleration for fixed point iteration. """
    res = []

    x = x0
    s = x.clone()
    t = torch.tensor(1., dtype=torch.float32)

    for k in range(2, max_iter):

        xnext = f(s)

        # acceleration

        tnext = 0.5*(1+torch.sqrt(1+4*t*t))

        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext

        res.append((x - s).norm().item()/(1e-5 + x.norm().item()))
        # if (res[-1] < tol):
        #     break

        # if k%10==0:
        #     n_ipt_temp = torch.unbind(iFFT(X[:,k%m].view_as(x0)), -1)[0]
        #     n_ipt_periter =  torch.cat([n_ipt_periter, n_ipt_temp[0,:]], dim=0)

    return x, res

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
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
        self.gamma = torch.nn.Parameter(torch.tensor(config['gamma_inti'], dtype=torch.float32), requires_grad=False)
        self.tau = torch.nn.Parameter(torch.tensor(config['tau_inti'], dtype=torch.float32), requires_grad=False) #torch.nn.Parameter(torch.tensor(config['tau_inti'], dtype=torch.float32, requires_grad=True))

        
    def forward(self, dObj, n_ipt:torch.tensor, n_y:torch.tensor, create_graph=False, strict=False, meas_list=None, gt=None):
        delta_g = dObj.grad(n_ipt, n_y)
        xSubD    = torch.abs(self.tau) * (self.dnn(n_ipt, create_graph, strict))
        xnext  = n_ipt - self.gamma * (delta_g.detach() + xSubD) # torch.Size([1, 1, H, W, 2])
        xnext[xnext<=0] = 0
        # if gt is not None:
        #     snr_iters = compare_snr(xnext, gt).item()
        #     print('psnr_iters:   ', snr_iters)
        return xnext

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, batch_size, **kwargs):
        super().__init__()

        self.f = f
        self.batch_size = batch_size
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.kwargs = kwargs
        self.dObj = CTClass()

    def forward(self, n_ipt, n_y=None, create_graph=True, strict=True, gt=None):
        # n_ipt_periter = n_ipt[0,:].detach()
        # compute forward pass and re-engage autograd tape
        z, self.forward_res = self.solver_img(lambda z : self.f(self.dObj, z, n_y, create_graph=False, strict=False, gt=gt), n_ipt, **self.kwargs)
        z = self.f(self.dObj, z.detach(), n_y, create_graph=create_graph, strict=strict)
        
        if create_graph:

            # set up Jacobian vector product (without additional forward calls)
            z0 = z.clone().detach().requires_grad_()
            # n_ipt_periter =  torch.cat((n_ipt_periter, z0[0,:]), dim=0)
            #f0 = self.f(self.dObj, z0, n_y, create_graph=create_graph, strict=create_graph)
            r0 = self.f.gamma * self.f.tau * self.f.dnn(z0, create_graph=create_graph, strict=strict)

            def backward_hook(grad):
                fTg = lambda y : y - self.f.gamma*self.dObj.fwd_bwd(y) - autograd.grad(r0, z0, y, retain_graph=True)[0] + grad
                g, self.backward_res = self.solver_grad(fTg,grad, **self.kwargs)
                return g
                
            # n_ipt_periter = torch.unsqueeze(n_ipt_periter, dim=1)
            
            z.register_hook(backward_hook)

        return [z, None, self.f.tau]#, loss_fp_dist

class Unfold_model(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""
    def __init__(self, dnn: nn.Module, config:dict):
        super(Unfold_model, self).__init__()
        self.f = dnn
        self.mode = config['mode'] 
        self.num_iter =  config['num_iter']
        self.dObj = CTClass()

    def forward(self, n_ipt:torch.tensor, n_y:torch.tensor, create_graph=False, strict=False, gt=None):
        tauList, gammaList = [], []
        
        if self.mode == 'onRED':
            # mode == 2 : implement online Red training 
            n_ipt_periter = n_ipt[0,:] # intial point
            for i in range(self.num_iter):
                n_ipt  = self.f(self.dObj, n_ipt, n_y, create_graph, strict, gt=gt)
                tauList.append(self.f.tau)
                n_ipt_periter =  torch.cat((n_ipt_periter, n_ipt[0,:]), dim=0)
            n_ipt_periter = torch.unsqueeze(n_ipt_periter,dim=1)    
        else:
            raise Exception("Unrecognized mode.")

        return [n_ipt, n_ipt_periter, self.f.tau]

class sgdnetCT(torch.nn.Module):
    """Train SGD-Net/Bach-Unfold with pixel loss"""
    def __init__(self, config, device):
        super(sgdnetCT, self).__init__()
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
            
            test_dataset = TestDataset(test_set['test_ipt'], 
                           test_set['test_gdt'], test_set['test_y'])

            self.test_dataLoader = DataLoader(test_dataset, 
                        batch_size=self.config['testing'].batch_size)
        elif train_set and valid_set is not None:

            train_dataset = TrainDataset(train_set['train_ipt'],
                       train_set['train_gdt'], train_set['train_y'])

            self.train_dataLoader = DataLoader(train_dataset, 
            batch_size=self.config['training'].batch_size, shuffle=True)

            valid_dataset = ValidDataset(valid_set['valid_ipt'], 
                            valid_set['valid_gdt'], valid_set['valid_y'])
            self.valid_dataLoader = DataLoader(valid_dataset, 
                         batch_size=self.config['validating'].batch_size)
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
            self.global_step = 0
            self.writer = SummaryWriter(log_dir=self.save_path+"/logs")
    
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
   
        self.URED = URED(self.dnn, self.config['unfold_model'], batch_size=fwd_set['batchAngles']).to(self.device)
        # self.network =  Unfold_model(self.URED, self.config['unfold_model']).to(self.device)

        self.network =  DEQFixedPoint(self.URED, anderson, anderson, 
                batch_size=fwd_set['batchAngles'], tol=1e-4, max_iter=self.config['unfold_model'].num_iter).to(self.device)

        if self.config['keep_training']:
            self.load_warmup()

        # Only DataParallel is provided at this moment.
        if self.config['num_gpus'] > 1:
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
    # ----------------------------------------
    # load pretrained for test
    # ----------------------------------------
    def load_test(self):
        load_path = os.path.join(self.config['root_path'], self.config['inference'].load_path)
        checkpoint = torch.load(load_path)['model_state_dict']

        try:
            self.network.load_state_dict(checkpoint,strict=True)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.network.load_state_dict(new_state_dict,strict=True)
        except Exception as err:
            print('Error occured at', err)

    # ----------------------------------------
    # load pretrained for warmup
    # ----------------------------------------
    def load_warmup(self):
        load_path = os.path.join(self.config['root_path'], self.config['keep_training'])
        checkpoint = torch.load(load_path)['model_state_dict']

        try:
            self.network.f.dnn.load_state_dict(checkpoint,strict=True)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.network.f.dnn.load_state_dict(new_state_dict,strict=True)
        except Exception as err:
            print('Error occured at', err)
    # # ----------------------------------------
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

        torch.save({
            'global_step':self.global_step,
            'epoch': epoch,
            'model_state_dict': self.dnn.state_dict()
        }, self.save_path+'/logs/dnn_red_epoch%d.pth'%(epoch))
        
    # ----------------------------------------
    # Test the pre-trained models
    # ----------------------------------------
    def test(self):
        
        self.load_test()
        count, sum_snr = 0, 0 
        self.network.eval()

        with torch.no_grad():

            for batch_ipt, batch_gdt, batch_y in tqdm(self.test_dataLoader):

                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_pre, _, _, batch_pre_iter = self.network(batch_ipt, batch_y)

                for minibatch in range(batch_pre.shape[0]):
                    snr = compare_snr(batch_pre[minibatch,...], batch_gdt[minibatch,...])
                    sum_snr = sum_snr + snr.item()
                    count = count + 1
        Avg_snr = sum_snr/count      
        print("==================================\n")
        print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
        print('Avg SNR: ', Avg_snr)
        print("\n==================================")
        print("")
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------    
    def train_optimize(self):
        
        for epoch in range(1, self.opt_train['unfold_epoch']):
            self.network.train()
            batch = 0
            avg_snr_training = 0
            avg_loss_training = 0
            if epoch < self.opt_train['unfold_lr_milstone']:
                current_lr = self.opt_train['unfold_lr']
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
            elif epoch >= self.opt_train['unfold_lr_milstone'] and epoch % 10 == 0:
                current_lr = current_lr *0.7
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
            # set learning rate
            print('learning rate %f' % current_lr)
            for batch_ipt, batch_gdt, batch_y in tqdm(self.train_dataLoader):
                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                batch_pre, batch_pre_iter, _  = self.network(batch_ipt, batch_y, create_graph=True, strict=True, gt=batch_gdt)
                # loss = loss_fn_l2(batch_pre, batch_gdt) #+ 0.5*loss_fn_l1(batch_pre, batch_gdt)
                loss = torch.mean(torch.pow(batch_pre - batch_gdt, 2))
                snr  = compare_snr(batch_pre, batch_gdt)
                avg_snr_training = snr.item() + avg_snr_training
                avg_loss_training = loss.item() + avg_loss_training            
                loss.backward()
                
                if self.global_step % 20==0:
                    grads_min = []
                    grads_max = []
                    for param in self.optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            grads_min.append(torch.min(param.grad))
                            grads_max.append(torch.max(param.grad))

                    grads_min = torch.min(torch.stack(grads_min,0))
                    grads_max = torch.max(torch.stack(grads_max,0))

                    self.writer.add_scalar('train/snr_iter', snr.item(), self.global_step)
                    self.writer.add_scalar('train/loss_iter', loss.item(), self.global_step)                    
                    self.writer.add_scalar('train/grad_iter_max_1', grads_max.item(), self.global_step)
                    self.writer.add_scalar('train/grad_iter_min_1', grads_min.item(), self.global_step)

                if epoch <= 15:
                    torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=1e-4)
                else:
                    torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=1e-5)  

                self.optimizer.step()
                batch += 1
                self.global_step += 1  
            with torch.no_grad():    
                avg_snr_training = avg_snr_training / batch
                avg_loss_training = avg_loss_training / batch                          
                Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=True, scale_each=True)
                Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=True, scale_each=True)
                # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=4, normalize=True, scale_each=True)
                self.writer.add_scalar('train/lr', current_lr, epoch)
                self.writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
                self.writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
                # self.writer.add_image('train/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
                self.writer.add_scalar('train/snr_epoch', avg_snr_training, epoch)
                self.writer.add_scalar('train/loss_epoch', avg_loss_training, epoch)            
                self.writer.add_histogram('train/histogrsm', batch_pre, epoch)
                # self.writer.add_histogram('train/tauList', np.asarray([t[0] for t in tauList]), epoch)#
            #####################################
            #               Valid               # 
            #####################################
            self.network.eval()
            sum_snr = 0
            sum_rsnr = 0 #Not necessary used for CT in this work 
            count = 0
            for batch_ipt, batch_gdt, batch_y in self.valid_dataLoader:
                batch_ipt = batch_ipt.to(self.device)
                batch_gdt = batch_gdt.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_pre, batch_pre_iter, _ = self.network(batch_ipt, batch_y, create_graph=False, strict=False)
                for minibatch in range(batch_pre.shape[0]):
                    snr = compare_snr(batch_pre[minibatch,...], batch_gdt[minibatch,...])
                    sum_snr = sum_snr + snr.item()
                    count = count + 1
                if snr.detach().cpu().numpy() < -300:
                    raise Exception("Wrong Initialization.")

            with torch.no_grad():    
                Avg_snr = sum_snr/count   
                self.writer.add_scalar('valid/snr',Avg_snr, epoch)
                self.writer.add_histogram('valide/histogrsm', batch_pre, epoch)
                print("==================================\n")
                print("epoch: [%d]" % epoch)
                print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
                print('snr: ', Avg_snr)
                print("\n==================================")
                print("")
                Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=True, scale_each=True)
                Img_ipt = utils.make_grid(batch_ipt, nrow=3, normalize=True, scale_each=True)
                Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=True, scale_each=True)
                # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=3, normalize=True, scale_each=True)
                # self.writer.add_image('valid/pre_iter', Img_pre_iter, epoch, dataformats='CHW')
                self.writer.add_image('valid/pre', Img_pre, epoch, dataformats='CHW')
                self.writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')
                self.writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')

            if epoch % self.opt_train['save_epoch'] == 0:
                self.save_model(epoch)
        self.writer.close()    
        ###############################################
        #                  End Training               #
        ###############################################
        