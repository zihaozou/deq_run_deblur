from datetime import datetime

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
###############################################
import torch
import hydra
import argparse
import numpy as np
from datetime import datetime

from utils.util import *
from models.sgdnet_Blur import sgdnetBlur
from models.sgdnet_CT import sgdnetCT
from models.sgdnet_IDT import sgdnetIDT
from models.sgdnet_MRI import sgdnetMRI

'''
Python 3.6
PyTorch 1.1.0~1.2.0
Windows 10 or Linux
Jiaming Liu (jiamingliu.jeremy@gmail.com)
github: https://github.com/wustl-cig/SGD-Net
If you have any question, please feel free to contact with me.
Jiaming Liu (e-mail: jiamingliu.jeremy@gmail.com)
by Jiaming Liu (16/Aug/2021)
'''

"""
# --------------------------------------------
testing/training code of SGD-Net for sparse-view CT and IDT in the paper
@article{liu2021sgd,
  title={Sgd-net: Efficient model-based deep learning with theoretical guarantees},
  author={Liu, J. and Sun, Y. and Gan, We. and Xu, X. and Wohlberg, B. and Kamilov, U. S.},
  journal={IEEE Transactions on Computational Imaging},
  year={2021},
  publisher={IEEE}
}
# --------------------------------------------
SGDNet/
├── model_zoos/
│   ├── sgdnet_ct_bs=60_full=120.pth
│   └── ...
└── sgdnet_run/
    ├── configs/
    │   └── config.yaml
    ├── data/
    │   ├── CT_ORG_train.csv
    │   └── CT_ORG_valid.csv
    │   └── CT_ORG_test.csv    
    ├── DataFidelities/
    │   ├── Dataclass.py
    │   ├── CTClass.py
    │   └── IDTClass.py
    ├── models/
    │   ├── loss_ssim.py
    │   ├── sgdnet_CT.py
    │   ├── sgdnet_IDT.py
    │   ├── unet.py
    │   └── unfold_network.py
    ├── main.py
    └── utils/
        ├── data_preprocess_CT.py
        ├── data_preprocess_IDT.py
        ├── net_mode.py
        └── util.py
# --------------------------------------------
"""

now = datetime.now()

def dataSet(config, device, set_mode='test'):

    if config['model_type'] == 'CT':
        from utils.data_preprocess_CT import data_preprocess
    elif config['model_type'] == 'IDT':    
        from utils.data_preprocess_IDT import data_preprocess
    elif config['model_type'] == 'MRI':    
        from utils.data_preprocess_MRI import data_preprocess    
    elif config['model_type'] == 'Blur':    
        from utils.data_preprocess_Blur import data_preprocess
    else:
        raise Exception("Unrecognized dataset.")

    train_set, valid_set, test_set = data_preprocess(config, device, set_mode)
    
    return train_set, valid_set, test_set

@hydra.main(config_name='configs/config')
def main(config):
    model_type = config['model_type']
    config = config[model_type]

    data_fidelity = {
        'CT': sgdnetCT,
        'IDT': sgdnetIDT,
        'MRI': sgdnetMRI,
        'Blur': sgdnetBlur
    }
    
    if config['root_path'] == 'None':
        root = os.path.dirname(os.path.abspath(__file__))
        config['code_path'] = root
        root = root.split('/')[0:-1]
        config['root_path'] = os.path.join('/',*root)
    save_path = config['root_path'] + '/experiements_%s/%s'%(model_type, str(now.strftime("%d-%b-%Y-%H-%M")))
    config['save_path'] = save_path
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if config['device'] == 'cuda' and config['multiGPU']:
        device = torch.device(f"cuda:{config['GPU_index'][0]}" if torch.cuda.is_available() else "cpu")
        config['num_gpus'] = torch.cuda.device_count()
    elif config['device'] == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config['num_gpus'] = 1
    else:
        device = torch.device('cpu')
        config['num_gpus'] = 0
    
    if config['set_mode']=='test':
        config['warm_up'] = False
        unfold = data_fidelity[model_type](config, device)
        _, _, test_set = dataSet(config, device, set_mode='test')
        unfold.init_state(test_set=test_set)
        unfold.test()
    elif config['set_mode']=='train':
        unfold = data_fidelity[model_type](config, device)
        train_set, valid_set, _ = dataSet(config, device, set_mode='train')
        unfold.init_state(train_set=train_set, valid_set=valid_set)
        unfold.train_optimize()
    else: 
        raise Exception("Only test/train are supported.") 

if __name__ == "__main__":
    print(torch.cuda.device_count())
    main()
