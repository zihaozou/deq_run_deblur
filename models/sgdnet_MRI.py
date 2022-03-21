from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm
from DataFidelities.MRIClass import MRIClass

from models.loss_ssim import SSIMLoss
from utils.net_mode import *
from utils.util import *

from DataFidelities.Dataclass import *

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

def nesterov(f, x0, max_iter=150):
    """ nesterov acceleration for fixed point iteration. """
    res = []
    imgs = []

    x = x0
    s = x.clone()
    t = torch.tensor(1., dtype=torch.float32)
    for k in range(max_iter):

        xnext = f(s)

        # acceleration

        tnext = 0.5*(1+torch.sqrt(1+4*t*t))

        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext

        # res.append((x - s).norm().item()/(1e-5 + x.norm().item()))
        # if (res[-1] < tol):
        #     break

    return x, res, None

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta = 1.0):
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
    return X[:,k%m].view_as(x0), res, None

class URED(nn.Module):

    def __init__(self, dObj, dnn: nn.Module, config:dict):
        #gamma_inti=3e-3, tau_inti=1e-1, batch_size=60, device='cuda:0'):

        super(URED, self).__init__()
        self.dnn = dnn
        self.dObj = dObj
        self.gamma = torch.tensor(config['gamma_inti'], dtype=torch.float32)
        self.tau = torch.tensor(config['tau_inti'], dtype=torch.float32)

    def denoise(self, n_ipt, create_graph=True, strict=True):
        denoiser = self.dnn(n_ipt, create_graph, strict)
        return denoiser

    def forward(self, n_ipt, n_y, mask, create_graph=False, strict=False, gt=None):
        # tau = self.tau if torch.abs(self.tau) - 0.03 <= 0 else 0.03
        delta_g = self.dObj.grad(n_ipt, n_y, mask)
        xSubD    = self.tau * (self.dnn(n_ipt, create_graph, strict))
        xnext  =  n_ipt - self.gamma * (delta_g.detach() + xSubD) # torch.Size([1, 1, H, W, 2])
        xnext[xnext<=0] = 0
        # snr_ = compare_snr(xnext, gt).item()
        # print(snr_)
        return xnext

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, **kwargs):
        super(DEQFixedPoint, self).__init__()
        self.f = f
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.kwargs = kwargs
        
    def forward(self, n_ipt, n_y, gt, create_graph=True, strict=True):
        tauList, gammaList,n_ipt_periter = None, None, None
        mask = self.f.dObj.init(gt)
        # compute forward pass and re-engage autograd tape
        z, self.forward_res, _ = self.solver_img(lambda z : self.f(z, n_y, mask, create_graph=False, strict=False, gt=gt), n_ipt, max_iter=100)
        z =  self.f(z.detach(), n_y, mask, create_graph=True, strict=True, gt=gt)
        # set up Jacobian vector product (without additional forward calls)
        if create_graph:
            z0 = z.clone().detach().requires_grad_()
            # f0 = self.f(z0, yStoc, emStoc, meas_list, create_graph=create_graph, strict=strict)
            r0 = self.f.gamma * self.f.tau * self.f.denoise(z0)
            def backward_hook(grad):
                # print(grad.shape)
                fTg = lambda y : y - self.f.gamma*self.f.dObj.fwd_bwd(y, mask)-autograd.grad(r0, z0, y, retain_graph=True)[0] + grad #
                g, self.backward_res, _ = self.solver_grad(fTg, grad, max_iter=50, **self.kwargs)
                return g
            z.register_hook(backward_hook)
        return z, tauList, gammaList, n_ipt_periter, None#, loss_fp_dist

class Unfold_model(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""
    def __init__(self, dnn: nn.Module, config):
        super(Unfold_model, self).__init__()
        self.f = dnn
        self.mode = config['unfold_model'].mode
        self.num_iter = config['unfold_model'].num_iter

    def forward(self, n_ipt, n_y=None, emParams=None, fgrad=None,  create_graph=True, strict=True):
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
                n_ipt = self.f(n_ipt, n_y, emParams, fgrad, create_graph, strict)
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

class sgdnetMRI(torch.nn.Module):
    """Train SGD-Net/Bach-Unfold with pixel loss"""
    def __init__(self, config, device):
        super(sgdnetMRI, self).__init__()
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
            self.emParams = test_set['emParams']          
        elif train_set and valid_set is not None:
            self.emParams = train_set['emParams']
            train_dataset = TrainDataset(train_set['train_ipt'],train_set['train_y'],train_set['train_gdt'])
            self.train_dataLoader = DataLoader(train_dataset, 
            batch_size=self.config['training'].batch_size, shuffle=True)
            valid_dataset = ValidDataset(valid_set['valid_ipt'], valid_set['valid_y'], valid_set['valid_gdt'])
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
            copytree(src=self.config['code_path'], dst=self.config['save_path'])
            self.define_loss()    
            self.define_optimizer()
            self.define_scheduler()
            self.global_step = 0
            self.writer = SummaryWriter(log_dir=self.save_path+"/logs")

    # ----------------------------------------
    # define unfold network (SGD-Net)
    # ----------------------------------------
    def define_model(self):
        self.dnn = net_model(self.config['cnn_model'])
        self.dnn.load_state_dict(torch.load('/export/project/jiaming.liu/Projects/Zihao/deq_run_radio/conv3.pt'))
        # self.dnn.apply(weights_init_kaiming)

        if self.set_mode == 'test':
            fwd_set = self.config['fwd_test']
        else:
            fwd_set = self.config['fwd_train']   
            if self.config['warm_up']:
                self.load_warmup()
        dObj = MRIClass(fwd_set, seed=self.config['seed'], emParams=self.emParams)
        self.URED = URED(dObj, self.dnn, self.config['unfold_model'])
        self.network =  DEQFixedPoint(self.URED, nesterov, anderson, tol=5e-5).to(self.device)
        if self.config['multiGPU']:
            gpu_ids = [4,5]
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
    # define optimizer
    # ----------------------------------------
    def define_scheduler(self):
        if self.config['training'].optimizer == "Adam":
            self.scheduler = None
        elif self.config['training'].optimizer == "SGD":
            if self.config['training'].scheduler == "Multistep":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,45,60,75,90,100,120,150], gamma=0.75)
            elif self.config['training'].scheduler == "CyclicLR":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                        step_size_up=1, step_size_down=58, base_lr=0.1, max_lr=0.2,mode='exp_range')

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
        load_path = os.path.join(self.config['root_path'], self.config['keep_training'].load_path)
        checkpoint = torch.load(load_path)['model_state_dict']
    
        try:
            self.dnn.load_state_dict(checkpoint,strict=True)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.dnn.load_state_dict(new_state_dict,strict=True)
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
    # update parameters and get loss
    # ----------------------------------------    
    def train_optimize(self):

        for epoch in range(1, self.opt_train['unfold_epoch']):
            self.network.train()
            batch = 0
            avg_snr_training = 0
            avg_rsnr_training = 0
            avg_loss_training = 0
            if epoch >= self.opt_train['unfold_lr_milstone'] and epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr'] *0.6
                # adjust learning rate
                print('current_lr: ', current_lr,'epoch: ', epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            # set learning rate
            print('learning rate %f' % self.optimizer.param_groups[0]['lr'])
            for batch_ipt, batch_y, batch_gdt in tqdm(self.train_dataLoader):
                batch_gdt = batch_gdt.to(self.device)
                batch_ipt = batch_ipt.to(self.device)
                self.optimizer.zero_grad()
                batch_pre, _, _, _,_ = self.network(batch_ipt, batch_y, batch_gdt,create_graph=False, strict=False)
                loss = self.lossfn(batch_pre, batch_gdt)
                snr  = psnr(batch_pre.detach().cpu().numpy(), batch_gdt.detach().cpu().numpy(),data_range=1)
                
                avg_snr_training = snr.item() + avg_snr_training
                avg_loss_training = loss.item() + avg_loss_training
                self.writer.add_scalar('train/psnr', snr.item(), self.global_step)
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                # self.writer.add_scalar('train/tau',tau[0].item(), self.global_step)
                loss.backward()

                if self.global_step % 5==0:
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

                torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value=5e-6)
                self.optimizer.step()

                if self.scheduler is not None and epoch < self.opt_train['unfold_lr_milstone']:
                    self.scheduler.step()

                batch += 1
                self.global_step += 1 

            with torch.no_grad():
                avg_snr_training = avg_snr_training / batch
                avg_loss_training = avg_loss_training / batch 
                batch_pre = torch.clamp(batch_pre,0,1)
                batch_gdt = torch.clamp(batch_gdt,0,1)         
                Img_pre = utils.make_grid(batch_pre, nrow=3, normalize=False, scale_each=True)
                Img_gdt = utils.make_grid(batch_gdt, nrow=3, normalize=False, scale_each=True)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_image('train/pre', Img_pre, epoch, dataformats='CHW')
                self.writer.add_image('train/gt', Img_gdt, epoch, dataformats='CHW')
                self.writer.add_scalar('train/snr_epoch', avg_snr_training, epoch)
                self.writer.add_scalar('train/loss_epoch', avg_loss_training, epoch)
                # self.writer.add_histogram('train/histogrsm', batch_pre, epoch)
            # #####################################
            # #               Valid               # 
            # #####################################
            self.network.eval()
            snr_avg = 0
            count = 0
            for batch_ipt, batch_y, batch_gdt in self.valid_dataLoader:
                batch_gdt = batch_gdt.to(self.device)
                batch_ipt = batch_ipt.to(self.device)
                batch_pre, _, _, _, _ = self.network(batch_ipt, batch_y, batch_gdt, create_graph=False)
                for i_test in range(batch_gdt.shape[0]):
                    snr_avg =  snr_avg + psnr(batch_pre[i_test,...].squeeze().detach().cpu().numpy(), batch_gdt[i_test,...].squeeze().detach().cpu().numpy(),data_range=1)
                    count = count + 1
        
            snr_avg = snr_avg / count
            self.writer.add_scalar('valid/psnr',snr_avg, epoch)
            self.writer.add_histogram('valide/histogrsm', batch_pre, epoch)
            print("==================================\n")
            print("epoch: [%d]" % epoch)
            print('batch_pre: ', torch.min(batch_pre), torch.max(batch_pre))
            print('psnr: ', snr_avg)
            print("\n==================================")
            print("")                
            batch_pre = torch.clamp(batch_pre,0,1)
            batch_gdt = torch.clamp(batch_gdt,0,1)           
            Img_pre = utils.make_grid(batch_pre, nrow=4, normalize=False, scale_each=True)
            Img_gdt = utils.make_grid(batch_gdt, nrow=4, normalize=False, scale_each=True)
            # Img_ipt = utils.make_grid(batch_ipt, nrow=4, normalize=False, scale_each=True)
            self.writer.add_image('valid/pre', Img_pre, epoch, dataformats='CHW')
            self.writer.add_image('valid/gt', Img_gdt, epoch, dataformats='CHW')
            # writer.add_image('valid/ipt', Img_ipt, epoch, dataformats='CHW')

            
            self.save_model(epoch)
        self.writer.close()
        #############################################
        #                 End Training              #
        ############################################# 
