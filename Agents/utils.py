import os
import json
import pickle
import imageio
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, root: str, tag: str, date=None, load_last=False) -> None:
        self.root = root
        self.tag = tag
        if load_last:
            path = os.path.join(self.runtime_root, self.tag)
            with open(path, 'rt') as file:
                data = json.load(file)
                self.date = data['date']
        else:
            self.date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') if \
                    date is None else date
        self.writer = None

    @property
    def runtime_root(self):
        path = os.path.join(self.root, 'runtime')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_runtime(self):
        # save this runtime args
        path = os.path.join(self.runtime_root, self.tag)
        with open(path, 'wt') as file:
            json.dump({
                'date': self.date,
            }, file)

    def save_tensorboard(self, tag, value, step):
        if self.writer is None:
            self.writer = SummaryWriter(os.path.join(
                self.root, 'tensorboard', f'{self.tag}-{self.date}/'))
        self.writer.add_scalar(tag, value, step)

    @property
    def model_root(self):
        path = os.path.join(self.root, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_model(self, model: nn.Module):
        path = os.path.join(self.model_root, f'{self.tag}-{self.date}.pth')
        torch.save(model.state_dict(), path)

    def load_model(self, model: nn.Module):
        path = os.path.join(self.model_root, f'{self.tag}-{self.date}.pth')
        model.load_state_dict(torch.load(path))

    def load_old_model(self, model: nn.Module, date):
        path = os.path.join(self.model_root, f'{self.tag}-{date}.pth')
        model.load_state_dict(torch.load(path))

    def save_old_model(self, model: nn.Module, date):
        path = os.path.join(self.model_root, f'{self.tag}-{date}.pth')
        torch.save(model.state_dict(), path)

    @property
    def param_root(self):
        path = os.path.join(self.root, 'argparse')
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def save_argparse(self, args):
        path = os.path.join(self.param_root, f'{self.tag}-{self.date}.json')
        with open(path, 'wt') as file:
            json.dump(vars(args), file, indent=4)

    def load_argparse(self, args):
        path = os.path.join(self.param_root, f'{self.tag}-{self.date}.json')
        args_dict = vars(args)
        with open(path, 'rt') as file:
            args_dict.update(json.load(file))

    @property
    def csv_root(self):
        path = os.path.join(self.root, 'csv')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_as_self_csv(self, result: dict, info):
        path = os.path.join(self.csv_root, f'{self.tag}-{info}-{self.date}.csv')
        sig = True
        for key in result.keys():
            if isinstance(result[key], list) or isinstance(result[key], np.ndarray):
                sig = False
                break

        if sig:
            df = pd.DataFrame(result, index=[0])
        else:
            df = pd.DataFrame(result)
        
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False)
        else:
            df.to_csv(path, header=True)

    def save_as_public_csv(self, result: dict, info):
        path = os.path.join(self.csv_root, info+'.csv')
        sig = True
        for key in result.keys():
            if isinstance(result[key], list) or isinstance(result[key], np.ndarray):
                sig = False
                break

        if sig:
            df = pd.DataFrame(result, index=[0])
        else:
            df = pd.DataFrame(result)
            
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False)
        else:
            df.to_csv(path, header=True)

    @property
    def npy_root(self):
        path = os.path.join(self.root, 'npy')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_npy(self, data, info: str):
        path = os.path.join(self.npy_root, f'{self.tag}-{info}-{self.date}.npy')
        np.save(path, data)
        
    @property
    def pack_root(self):
        path = os.path.join(self.root, 'pack')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_pack(self, result, info):
        path = os.path.join(self.pack_root, f'{self.tag}-{info}-{self.date}.pkl')
        with open(path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    @property
    def gif_root(self):
        path = os.path.join(self.root, 'gif')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_frames_as_gif(self, frames, info):
        path = os.path.join(self.gif_root, f'{self.tag}-{info}-{self.date}.gif')
        patch = plt.imshow(frames[0])
        plt.axis('off')
            
        imageio.mimsave(path, frames, 'GIF', duration=1/60)


class ReplayBuffer:
    def __init__(self, buf_size) -> None:
        self.size = buf_size
        self.len = 0
        self.idx = 0
        self.attrs = {}
        self.random = np.random.RandomState()

    def store(self, **argkv):
        """Store the exp"""
        if len(self.attrs) == 0:
            for k, v in argkv.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    dtype = type(v[0])
                elif isinstance(v, np.ndarray):
                    dtype = v.dtype
                elif isinstance(v, torch.Tensor):
                    dtype = object
                else:
                    dtype = type(v)

                self.attrs[k] = dtype
                setattr(self, k, np.ndarray(
                    (self.size,) + np.array(v).shape if dtype != object else (self.size,), 
                    dtype=dtype))

        for k, v in argkv.items():
            if k not in self.attrs:
                raise KeyError(
                    f"The buffer only has keys: {self.attrs}; but want to store {k}!")
            getattr(self, k)[self.idx] = v
        self.idx = (self.idx + 1) % self.size
        self.len = min(self.size, self.len + 1)

    def clear(self):
        """Clear the buffer."""
        self.len = 0
        self.idx = 0

    def sample(self, batch_size=-1, shuffle=True) -> dict:
        """
        Sample exp from the buffer. \n
        Batch_size -1 for get all exp; \n
        Shuffle True for shuffle the exp it sampled.
        """
        if batch_size < 0:
            batch_size = self.len
        if batch_size > self.len:
            raise RuntimeError(f"ReplayBuffer only has {self.len} exp, but it want to get {batch_size}!")
        
        idx = self.random.choice(self.len, batch_size, replace=False)
        
        if not shuffle:
            idx = sorted(idx)
        return {
            k: getattr(self, k)[idx]
            for k in self.attrs
        }

    def seed(self, seed):
        """The random seed for sample func."""
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.len

    def isfull(self):
        """When the num of exp is equal to the buffer size."""
        return self.len == self.size
    
    def isempty(self):
        """When the num of exp is zero."""
        return self.len == 0

if __name__ == '__main__':
    buf = ReplayBuffer(2)
    assert buf.isempty()

    buf.store(state=torch.rand([1, 3]), next_state=[1,2,3], action=1, reward=0.5)
    assert not buf.isfull()
    buf.store(state=torch.rand([1, 3]), next_state=[2,3,1], action=2, reward=0.7)
    assert buf.isfull()
    for k, v in buf.sample(2).items():
        print(k)
        print(v)
    print("Cur num of exp:", len(buf))

    try:
        buf.store(states=1)
    except KeyError:
        print('KeyError test pass.')

    # print(buf.sample(4))