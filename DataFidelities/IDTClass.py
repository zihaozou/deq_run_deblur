import h5py
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset

from pathlib import Path

def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([res_real, res_imag], -1)

###################
# Read Images
###################
def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)

class IDTClass(nn.Module):

    def __init__(self):
        super(IDTClass, self).__init__()

    def fgrad(self, f_ipt, f_y, emParamsTrain_cuda):
        f_ph  = f_ipt
        f_ab  = torch.zeros_like(f_ph)

        gradPhList = []
        for i in range(f_ph.shape[0]):
            gradPhList.append(self.gradPh(f_ph[i], f_ab[i], emParamsTrain_cuda, f_y[i]))

        f_grad = torch.stack(gradPhList, 0)

        return f_grad
    
    def fgrad_SGD(self, f_ipt, f_y, meas_list, emParamsTrain_cuda):
        f_ph  = f_ipt
        f_ab  = torch.zeros_like(f_ph)

        gradPhList = []
        for i in range(f_ph.shape[0]):
            gradPhList.append(self.gradPhStoc(f_ph[i], f_ab[i], meas_list, emParamsTrain_cuda, f_y[i]))

        f_grad = torch.stack(gradPhList, 0)

        return f_grad

    def fwd_bwd(self, f_ipt, emStoc=None) :

        # sub =  torch.randperm(emParamsTrain_cuda['NBFkeep']).tolist()
        # meas_list = sub[0:meas_list]
        # meas_list.sort()

        # emStoc = {}
        # emStoc['NBFkeep'] = len(meas_list)
        # emStoc['Hreal'] = emParamsTrain_cuda['Hreal'][meas_list,:].to(f_ipt.device)  
        # # emStoc['Himag'] = emParams['Himag'][meas_list,:]

        f_ph  = f_ipt
        f_ab  = torch.zeros_like(f_ph)
        gradPhList = []
        with torch.no_grad():
            for i in range(f_ph.shape[0]):
                gradPhList.append(IDTClass.ftran(IDTClass.fmult(f_ph[i], f_ab[i], emStoc), emStoc, 'Ph'))
            f_grad = torch.stack(gradPhList, 0)
        return f_grad

    @staticmethod
    def gradPh(ph, ab, emParams, y):
        z = IDTClass.fmult(ph, ab, emParams)
        g = IDTClass.ftran(z - y, emParams, 'Ph')
        return g

    @staticmethod
    def gradAb(ph, ab, emParams, y):
        z = IDTClass.fmult(ph, ab, emParams)
        g = IDTClass.ftran(z - y, emParams, 'Ab')
        return g

    @staticmethod
    def gradPhStoc(ph, ab, meas_list, emStoc, yStoc):  
        zStoc = IDTClass.fmult(ph, ab, emStoc)
        g = IDTClass.ftran(zStoc - yStoc, emStoc, 'Ph')
        return g

    @staticmethod
    def fmult(ph, ab, emParams):

        ph = ph.unsqueeze_(0)
        ph = ph.expand(size=(emParams['NBFkeep'], ) + ph.shape[1:])

        ab = ab.unsqueeze_(0)
        ab = ab.expand(size=(emParams['NBFkeep'], ) + ab.shape[1:])
        z = complex_multiple_torch(emParams['Hreal'], ph) # + complex_multiple_torch(emParams['Himag'], ab)
        return z

    @staticmethod
    def ftran(z, emParams, which):
        assert which in ['Ph', 'Ab'], "Error in which"
        if which == 'Ph':
            Hreal = emParams['Hreal']
            Hreal_real, Hreal_imag = torch.unbind(Hreal, -1)
            Hreal_imag = -Hreal_imag

            Hreal = torch.stack([Hreal_real, Hreal_imag], -1)

            x = torch.sum(complex_multiple_torch(Hreal, z), 0)
        else:
            Himag = emParams['Himag']
            Himag_real, Himag_imag = torch.unbind(Himag, -1)
            Himag_imag = -Himag_imag
            Himag = torch.stack([Himag_real, Himag_imag], -1)

            x = torch.sum(complex_multiple_torch(Himag, z), 0)
        x = x / emParams['NBFkeep']
        return x

class index_choose_():
    def __init__(self, index_sets):

        if len(index_sets) != 0:

            self.set_indx = index_sets['angle_index']
            self.used_index = index_sets['used_index'] - 1
            self.angle_lst = []

            for i in range(self.set_indx.shape[0]):
                set_indx_temp = self.set_indx[i]
                angle_lst_sub = []

                for j in range(set_indx_temp.shape[0]):
                    if set_indx_temp[j] != 0:
                        angle_lst_sub.append(set_indx_temp[j] - 1)

                self.angle_lst.append(angle_lst_sub) 
        else:
            pass        

    def get_subset_radial(self, batch_size=3):

        sub = np.random.choice(len(self.angle_lst), batch_size, replace=False)
        # print(sub)
        sub_list = []
        for i in sub:
            sub_list = sub_list + self.angle_lst[i] 
        # print(len(sub_list))
        seen = set()
        uniq = []
        for x in sub_list:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        uniq.sort()
        sub_list = uniq
        # print(sub_list)
        sub_list_all = [np.where(self.used_index == i) for i in sub_list]

        sub_list_all = np.concatenate(sub_list_all, 0).squeeze()

        return sub_list_all

    @staticmethod
    def get_subset_uniform(NBFkeep=92,  batch_size=30, num_div=5):

        sub =  torch.randperm(NBFkeep//num_div)[0:batch_size//num_div]
        sub, _ = torch.sort(sub)
        meas_list = torch.cat([sub + i*NBFkeep//num_div for i in range(num_div)]).tolist()
        meas_list.sort()
        return meas_list

    @staticmethod
    def get_subset_random(NBFkeep=92,  batch_size=30):
        sub =  torch.randperm(NBFkeep).tolist()
        meas_list = sub[0:batch_size]
        meas_list.sort()
        return meas_list

class TrainDataset(Dataset):

    def __init__(self,
                       train_y:torch.Tensor, 
                       train_gdt:torch.Tensor,
                ):
          
        super(TrainDataset, self).__init__()
        self.train_y = train_y
        self.train_gdt = train_gdt

    def __len__(self):
        return self.train_gdt.shape[0]

    def __getitem__(self, item):
        return self.train_y[item], self.train_gdt[item]

class ValidDataset(Dataset):

    def __init__(self,
                       valid_y:torch.Tensor, 
                       valid_gdt:torch.Tensor,
                       ):
          
        super(Dataset, self).__init__()
        self.valid_y = valid_y
        self.valid_gdt = valid_gdt
    def __len__(self):
        return self.valid_gdt.shape[0]

    def __getitem__(self, item):
        return self.valid_y[item], self.valid_gdt[item]

class TestDataset(Dataset):

    def __init__(self, test_ipt:torch.Tensor, 
                       test_gdt:torch.Tensor,
                       ):
          
        super(Dataset, self).__init__()
        self.test_ipt = test_ipt
        self.test_gdt = test_gdt
    def __len__(self):
        return self.test_gdt.shape[0]

    def __getitem__(self, item):
        return self.test_ipt[item], self.test_gdt[item]

class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        train_gdt = self.get_data("train_gdt", index)

        if self.transform:
            train_gdt = self.transform(train_gdt)
        else:
            train_gdt = torch.from_numpy(train_gdt)

        # get label
        train_ipt = self.get_data("train_ipt", index)
        train_ipt = torch.from_numpy(train_ipt)

        train_y = self.get_data("train_y", index)
        train_y = torch.from_numpy(train_y)

        return train_ipt, train_gdt, train_y

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for dname, ds in h5_file.items():
                # for dname, ds in group.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)
                
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]