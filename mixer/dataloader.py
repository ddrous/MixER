from typing import Tuple
from ._utils import *
from abc import abstractmethod

import pandas as pd
import torch
from torchvision.transforms import transforms
from torch.utils import data
from PIL import Image



class DataLoader:
    """
    A base class generator of generators for general-purpose meta-learning regression tasks.
    """
    def __init__(self, 
                 data_path, 
                 envs_batch_size=250, 
                 envs_shuffle=True, 
                 shots_batch_size=1,
                 shots_shuffle=False, 
                 data_split="train", 
                 key=None):

        self.data_path = data_path
        self.envs_batch_size = envs_batch_size
        self.envs_shuffle = envs_shuffle

        self.adaptation = data_split in ["adapt", "test"]
        self.shots_batch_size = shots_batch_size
        self.shots_shuffle = shots_shuffle

        ## Define in the child class
        self.nb_batches = None

        if shots_batch_size <= 0 or shots_batch_size <=0 :
            raise ValueError("A batch size must be greater than 0.")

        self.key = key
        if (self.envs_shuffle or self.shots_shuffle) and self.key is None:
            raise ValueError("Shuffling the dataset requires a key.")


    @abstractmethod
    def sample_environments(self, key, batch_id, nb_envs):
        """ Provides a stateless way to sample a batch of environments """
        pass

    @abstractmethod
    def __iter__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Loads, transforms and yields a batch of environments """
        pass

    @abstractmethod
    def __len__(self):
        """ Total number of environments / envs_batch size. """
        pass

    def __str__(self) -> str:
        return f"Dataloader properties: \n" + \
                f"Total number of environments: {len(self)} \n" + \
                f"Batch size (envs): {self.envs_batch_size} \n" + \
                f"Number of points per environment = batch size (datapoints): {self.shots_batch_size} \n" + \
                f"Input dimension: {self.input_dim} \n" + \
                f"Output dimension: {self.output_dim} \n"





class CelebADataLoader(data.DataLoader):
    """
    A celeb a dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 envs_batch_size=250, 
                 envs_shuffle=True, 
                 shots_batch_size=100,
                 shots_shuffle=False, 
                 data_split="train",
                 resolution=(32, 32),
                 order_pixels=False,
                 key=None):

        super().__init__(data_path, 
                        envs_batch_size, 
                        envs_shuffle, 
                        shots_batch_size, 
                        shots_shuffle, 
                        data_split, 
                        key)

        self.input_dim = 2
        self.output_dim = 3
        self.img_size = (*resolution, self.output_dim)
        self.order_pixels = order_pixels
        ## Read the partitioning file: train(0), val(1), test(2)

        partitions = pd.read_csv(self.data_path+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        if data_split in ["train"]:
            self.files = partitions[partitions['partition'] == 0]['filename'].values
        elif data_split in ["val"]:
            self.files = partitions[partitions['partition'] == 1]['filename'].values
        elif data_split in ["test"]:
            self.files = partitions[partitions['partition'] == 2]['filename'].values
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        ## A list of MVPs images (or the worst during self-modulation) - Useful for active learning
        # self.mvp_files = self.files

        self.total_envs = len(self.files)
        if self.total_envs == 0:
            raise ValueError("No files found for the split.")
        if envs_batch_size > self.total_envs:
            raise ValueError(f"Envs batch size must be less than the total number of environments")

        self.total_pixels = self.img_size[0] * self.img_size[1]
        if shots_batch_size > self.total_pixels:
            raise ValueError(f"Few shots batch size must be less than the total number of pixels")

        ## Ssee CAVIA code: https://github.com/lmzintgraf/cavia)
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                            transforms.Resize((self.img_size[0], self.img_size[1]), Image.LANCZOS),
                                            transforms.ToTensor(),
                                            ])

        ## Batch bookeeping
        self.nb_batches = np.ceil(self.total_envs / self.envs_batch_size).astype(int)
        self.remainder = self.total_envs % self.envs_batch_size
        self.curr_batch_id = 0


    def get_image(self, filename) -> torch.Tensor:
        img_path = os.path.join(self.data_path+"img_align_celeba/", filename)
        img = self.transform(img_path).float()
        img = img.permute(1, 2, 0)
        return img

    def sample_pixels(self, key, img) -> Tuple[np.ndarray, jnp.ndarray]:
        total_pixels = self.img_size[0] * self.img_size[1]

        if self.order_pixels:
            flattened_indices = jnp.arange(self.shots_batch_size)
        else:
            flattened_indices = jax.random.choice(key=key, a=total_pixels, shape=(self.shots_batch_size,), replace=False)

        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coordinates = np.vstack((x, y)).T
        coords = torch.from_numpy(coordinates).float()
        normed_coords = (coords / torch.Tensor(self.img_size[:2])).numpy()

        pixel_values = img[coords[:, 0].long(), coords[:, 1].long(), :].numpy()

        return normed_coords, pixel_values


    def sample_environments(self, key, batch_id, nb_envs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Sample a batch of environments """

        X = np.zeros((nb_envs, self.shots_batch_size, self.input_dim))
        Y = np.zeros((nb_envs, self.shots_batch_size, self.output_dim))

        if self.envs_shuffle:
            sample_idx = jax.random.choice(key=key, a=self.total_envs, shape=(nb_envs,))
            sampled_files = self.files[sample_idx]
        else:
            f_start = batch_id*self.envs_batch_size
            f_end = min([(batch_id+1)*self.envs_batch_size, self.total_envs])
            sampled_files = self.files[f_start:f_end]

        pixel_keys = jax.random.split(key, num=nb_envs)
        for e, img_name in enumerate(sampled_files):
            img = self.get_image(img_name)
            normed_coords, pixel_values = self.sample_pixels(pixel_keys[e], img)
            X[e, :, :] = normed_coords
            Y[e, :, :] = pixel_values

        return jnp.array(X), jnp.array(Y)


    def make_batch(self):

        ## Sample a batch of environments
        if self.curr_batch_id == self.nb_batches-1 and self.remainder != 0:
            X, Y = self.sample_environments(self.key, self.curr_batch_id, self.remainder)
        else:
            X, Y = self.sample_environments(self.key, self.curr_batch_id, self.envs_batch_size)

        ##  Usefull when pixels are ordered
        if self.shots_shuffle:
            X = jax.random.permutation(self.key, X, axis=1)
            Y = jax.random.permutation(self.key, Y, axis=1)

        ## Update the state of the dataloader
        self.key, _ = jax.random.split(self.key)
        self.curr_batch_id += 1

        return X, Y


    def __iter__(self):
        self.curr_batch_id = 0
        return self
        # while self.curr_batch_id < self.nb_batches:
        #     yield self.make_batch()

    def __next__(self):
        if self.curr_batch_id < self.nb_batches:
            return self.make_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.total_envs



































## Todo. It might be better to write a new DataLoader from the Torch class altogether (See NumpyLoader)
def collate_to_jax(batch):
    xs, ys = zip(*batch)
    return jnp.array(xs), jnp.array(ys)


class CelebADataset(DataLoader):
    """
    A celeb a dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 resolution=(32, 32),
                 order_pixels=False,
                 seed=None):

        # ## Set seed
        # if seed is not None:
        #     np.random.seed(seed)

        if num_shots <= 0:
            raise ValueError("Number of shots must be greater than 0.")
        elif num_shots > resolution[0]*resolution[1]:
            raise ValueError("Number of shots must be less than the total number of pixels.")
        self.nb_shots = num_shots

        self.input_dim = 2
        self.output_dim = 3
        self.img_size = (*resolution, self.output_dim)
        self.order_pixels = order_pixels
        ## Read the partitioning file: train(0), val(1), test(2)

        self.data_path = data_path
        partitions = pd.read_csv(self.data_path+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        if data_split in ["train"]:
            self.files = partitions[partitions['partition'] == 0]['filename'].values
        elif data_split in ["val"]:
            self.files = partitions[partitions['partition'] == 1]['filename'].values
        elif data_split in ["test"]:
            # self.files = partitions[partitions['partition'] == 2]['filename'].values

            ## To get the translation-equivariance img in front of the test set (incl. Ellen selfie)
            self.files = partitions[(partitions['partition'] == 2) | (partitions['partition'] == 3)]['filename'].values
            self.files = np.concatenate((self.files[-1:], self.files[:-1]))

        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        if data_split in ["train", "val"]:
            self.adaptation = False
        elif data_split in ["test"]:
            self.adaptation = True
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        ## A list of MVPs images (or the worst during self-modulation) - Useful for active learning
        # self.mvp_files = self.files

        self.total_envs = len(self.files)
        if self.total_envs == 0:
            raise ValueError("No files found for the split.")

        self.total_pixels = self.img_size[0] * self.img_size[1]

        ## Ssee CAVIA code: https://github.com/lmzintgraf/cavia)
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                            transforms.Resize((self.img_size[0], self.img_size[1]), Image.LANCZOS),
                                            transforms.ToTensor(),
                                            ])

    def get_image(self, filename) -> torch.Tensor:
        img_path = os.path.join(self.data_path+"img_align_celeba/", filename)
        img = self.transform(img_path).float()
        img = img.permute(1, 2, 0)
        return np.array(img)

    def sample_pixels(self, img) -> Tuple[np.ndarray, jnp.ndarray]:        ## TODO: Stay in torch throughout this function!
        total_pixels = self.img_size[0] * self.img_size[1]

        if self.order_pixels:
            flattened_indices = np.arange(self.nb_shots)
        else:
            flattened_indices = np.random.choice(total_pixels, size=self.nb_shots, replace=False)

        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coords = np.vstack((x, y)).T
        normed_coords = (coords / np.array(self.img_size[:2]))

        pixel_values = img[coords[:, 0], coords[:, 1], :]

        return normed_coords, pixel_values

    def set_seed_sample_pixels(self, seed, idx):
        np.random.seed(seed)
        # np.random.set_state(seed)
        img = self.get_image(self.files[idx])
        return self.sample_pixels(img)


    def __getitem__(self, idx):
        img = self.get_image(self.files[idx])
        normed_coords, pixel_values = self.sample_pixels(img)
        return normed_coords, pixel_values


    def __len__(self):
        return self.total_envs











def numpy_collate(batch):
  return jax.tree.map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    
    self.num_batches = np.ceil(len(dataset) / batch_size).astype(int)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
  
















class SinusoidDataset:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    # def __init__(self, meta_tain=True, support_set=True):
    def __init__(self, num_envs, num_shots, adaptation=False):

        self.num_inputs = 1
        self.num_outputs = 1

        self.amplitude_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]

        self.input_range = [-5, 5]

        # if meta_tain:
        #     self.total_envs = 16
        # else:   ## Meta-test
        #     self.total_envs = 32

        # if support_set:
        #     self.num_shots = 4
        # else:   ## Query set
        #     self.num_shots = 1

        self.num_shots = num_shots
        self.total_envs = num_envs

        self.adaptation = adaptation

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def sample_task(self):
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
        return self.get_target_function(amplitude, phase)

    @staticmethod
    def get_target_function(amplitude, phase):
        def target_function(x):
            if isinstance(x, torch.Tensor):
                return torch.sin(x - phase) * amplitude
            else:
                return np.sin(x - phase) * amplitude

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False):

        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)

        target_functions = []
        for i in range(num_tasks):
            target_functions.append(self.get_target_function(amplitude[i], phase[i]))

        if return_specs:
            return target_functions, amplitude, phase
        else:
            return target_functions

    def __getitem__(self, idx):     ## TODO Idx doesn't matter here. Check effect of seed as well.
        if not self.adaptation:
            np.random.seed(idx)
            # torch.manual_seed(idx)
        else:
            np.random.seed(np.iinfo(np.int32).max - idx)
            # torch.manual_seed(np.iinfo(np.int32).max - idx)

        target_func = self.sample_tasks(1, return_specs=False)[0]
        inputs = self.sample_inputs(self.num_shots)
        return inputs, target_func(inputs)

    def __len__(self):
        return self.total_envs








class DynamicsDataset:
    """
    For all dynamics tasks as in Kirchmeyer et al. 2022
    """

    # def __init__(self, meta_tain=True, support_set=True):
    def __init__(self, data_dir, num_shots=-1, skip_steps=1, adaptation=False, return_t_eval=False):

        self.data_dir = data_dir
        self.skip_steps = skip_steps
        # self.adaptation = data_dir.find("adapt") != -1 or data_dir.find("ood") != -1
        self.adaptation = adaptation
        self.return_t_eval = return_t_eval

        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not found at {data_dir}")

        self.dataset, self.t_eval = raw_data['X'][...,::self.skip_steps,:], raw_data['t'][::skip_steps]

        datashape = self.dataset.shape
        self.total_envs = datashape[0]

        if num_shots is None or num_shots == -1:
            num_shots = datashape[1]
        self.num_shots = num_shots
        if num_shots > datashape[1]:
            raise ValueError("Number of shots must be less than the total number of trajectories")

        self.num_steps = datashape[2]
        self.data_size = datashape[3]


    def __getitem__(self, idx):
        inputs = self.dataset[idx, :, 0, :]
        outputs = self.dataset[idx, :, :, :]

        if self.return_t_eval:
            return (inputs, self.t_eval), outputs
        else:
            return inputs, outputs

    def __len__(self):
        return self.total_envs



class MixerDynamicsDataset(DynamicsDataset):
    """
    For all dynamics tasks as in Kirchmeyer et al. 2022, always returns the evaluation time stamps
    """
    def __init__(self, data_dir, num_shots=-1, skip_steps=1, adaptation=False):
        super().__init__(data_dir, num_shots, skip_steps, adaptation, return_t_eval=True)




class ODEBenchDataset:
    """
    For all dynamics tasks as in the ODEBench paper
    """

    def __init__(self, data_dir, norm_consts=None, num_shots=-1, skip_steps=5, adaptation=False, traj_prop_min=1.0):

        self.data_dir = data_dir
        self.skip_steps = skip_steps
        self.adaptation = adaptation

        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset, t_eval = raw_data['X'][...,::self.skip_steps,:], raw_data['t'][..., ::skip_steps]

        n_odes, n_envs_per_ode, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape
        n_odes, n_timesteps = t_eval.shape
        ## Load the bounds and normalise the dataset
        if norm_consts is None:
            ##  TODO remember to build a Network that predicts this scaling based on the context
            # raise ValueError("Normalisation constants must be provided, unique per environment across train and test")
            pass
        else:
            print("Normalising the dataset (one constant per family of environments)")
            norm_consts = np.load(norm_consts)
            dataset = dataset / norm_consts

        ## Merge the two first dataset dimensions
        self.dataset = dataset.reshape(n_odes*n_envs_per_ode, n_trajs_per_env, n_timesteps, n_dimensions)
        # print("SO here's the dataset shape:", self.dataset.shape)
        ## Duplicate t_eval for each environment
        self.t_eval = np.repeat(t_eval, n_envs_per_ode, axis=0)

        # # ## Ignore the first 16*2 TODO temporary
        # self.dataset = self.dataset[5*2:, :, :, :]
        # self.t_eval = self.t_eval[5*2:, :]

        self.total_envs = n_odes*n_envs_per_ode

        # ## Normalise the dynamics between -1 and 1, for each environment - TODO remember to build a Network that predicts this scaling based on the context
        # max_vals = np.max(np.abs(self.dataset), axis=(1,2), keepdims=True)
        # self.dataset = self.dataset / max_vals

        if num_shots is None or num_shots == -1:
            num_shots = n_trajs_per_env
        self.num_shots = num_shots
        if num_shots > n_trajs_per_env:
            raise ValueError("Number of shots must be less than the total number of trajectories")

        if traj_prop_min < 0 or traj_prop_min > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop_min = traj_prop_min

        self.num_steps = n_timesteps
        self.data_size = n_dimensions

    def __getitem__(self, idx):
        inputs = self.dataset[idx, :self.num_shots, 0, :]
        outputs = self.dataset[idx, :self.num_shots, :, :]
        t_evals = self.t_eval[idx]

        if self.traj_prop_min == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_evals), outputs

        else:
            ### SAMPLING APPROACH ###
            ## Sample a start and end time step for the task, and interpolate to produce new timesteps
            ### The minimum distance between the start and finish is min_len
            traj_len = t_evals.shape[0]
            new_traj_len = traj_len ## We always want traj_len samples
            min_len = int(traj_len * self.traj_prop_min)
            start_idx = np.random.randint(0, traj_len - min_len)
            end_idx = np.random.randint(start_idx + min_len, traj_len)

            ts = t_evals[start_idx:end_idx]
            trajs = outputs[:, start_idx:end_idx, :]
            new_ts = np.linspace(ts[0], ts[-1], new_traj_len)
            new_trajs = np.zeros((self.num_shots, new_traj_len, self.data_size))
            for i in range(self.num_shots):
                for j in range(self.data_size):
                    new_trajs[i, :, j] = np.interp(new_ts, ts, trajs[i, :, j])

            return (new_trajs[:,0,:], new_ts), new_trajs


    def __len__(self):
        return self.total_envs
        # return 5*1      ### TODO this is temporary






class TimeSeriesDataset:
    """
    For any time series dataset, from which the others will inherit
    """
    def __init__(self, dataset, t_eval, adaptation=False, traj_prop_min=1.0, use_full_traj=True):

        self.dataset = dataset
        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape
        self.t_eval = t_eval
        self.total_envs = n_envs

        if traj_prop_min < 0 or traj_prop_min > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop_min = traj_prop_min

        self.num_steps = n_timesteps
        self.data_size = n_dimensions
        self.num_shots = 1
        self.use_full_traj = use_full_traj      ## If True, we use the full trajectory, else we only return the initial condition
        self.adaptation = adaptation

    def __getitem__(self, idx):
        if self.use_full_traj:
            inputs = self.dataset[idx, :self.num_shots, :, :]
        else:
            inputs = self.dataset[idx, :self.num_shots, 0, :]
        outputs = self.dataset[idx, :self.num_shots, :, :]
        t_evals = self.t_eval[idx]

        if self.traj_prop_min == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_evals), outputs
        else:
            ### SAMPLING APPROACH ###
            ## Sample a start and end time step for the task, and interpolate to produce new timesteps
            ### The minimum distance between the start and finish is min_len
            traj_len = t_evals.shape[0]
            new_traj_len = traj_len ## We always want traj_len samples
            min_len = int(traj_len * self.traj_prop_min)
            start_idx = np.random.randint(0, traj_len - min_len)
            end_idx = np.random.randint(start_idx + min_len, traj_len)

            ts = t_evals[start_idx:end_idx]
            trajs = outputs[:, start_idx:end_idx, :]
            new_ts = np.linspace(ts[0], ts[-1], new_traj_len)
            new_trajs = np.zeros((self.num_shots, new_traj_len, self.data_size))
            for i in range(self.num_shots):
                for j in range(self.data_size):
                    new_trajs[i, :, j] = np.interp(new_ts, ts, trajs[i, :, j])

            if self.use_full_traj:
                return (new_trajs[:,:,:], new_ts), new_trajs
            else:
                return (new_trajs[:,0,:], new_ts), new_trajs

    def __len__(self):
        return self.total_envs


class EpilepsyDataset(TimeSeriesDataset):
    """
    For the Epilepsy2 dataset from Time Series Classification
    """

    def __init__(self, data_dir, skip_steps=-1, adaptation=False, traj_prop_min=1.0, use_full_traj=True):
        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data['signal'][:, None, ::skip_steps, None]

        n_envs, _, n_timesteps, _ = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)
        t_eval = np.repeat(t_eval[None,:], n_envs, axis=0)

        super().__init__(dataset, t_eval, adaptation, traj_prop_min, use_full_traj)




class CrowdsourceDataset(TimeSeriesDataset):
    """
    For the EEG crowdsource dataset from Navid and Amarpal
    """

    def __init__(self, data_dir, data_split="train", skip_steps=-1, adaptation=False, traj_prop_min=1.0, use_full_traj=True):
        try:
            raw_data = np.load(data_dir+"Crowdsource.npy", allow_pickle=True).item()
        except:
            raise ValueError(f"Data not found at {data_dir}")

        key = data_split+"_data" if data_split in ["train", "val"] else "test_data"

        dataset = raw_data[key].transpose(0, 2, 1)[:, None, ::skip_steps, :]

        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)
        t_eval = np.repeat(t_eval[None,:], n_envs, axis=0)

        super().__init__(dataset, t_eval, adaptation, traj_prop_min, use_full_traj)


class TrendsDataset(TimeSeriesDataset):
    """
    For the synthetic control dataset from Time Series Classification
    """

    def __init__(self, data_dir, skip_steps=-1, adaptation=False, traj_prop_min=1.0, use_full_traj=True):
        try:
            time_series = []
            with open(data_dir+"synthetic_control.data", 'r') as f:
                for line in f:
                    time_series.append(list(map(float, line.split())))
            raw_data = np.array(time_series, dtype=np.float32)
            raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)        ## normalise the dataset
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data[:, None, ::skip_steps, None]

        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)
        t_eval = np.repeat(t_eval[None,:], n_envs, axis=0)

        super().__init__(dataset, t_eval, adaptation, traj_prop_min, use_full_traj)


class SleepDataset(TimeSeriesDataset):
    """
    For the EEG crowdsource dataset from Navid and Amarpal
    """

    def __init__(self, data_dir, data_split="train", skip_steps=-1, adaptation=False, traj_prop_min=1.0, use_full_traj=True):
        import h5py

        try:
            with h5py.File(data_dir+'dataset.h5', "r") as f:
                if data_split in ["train", "test"]:
                    raw_data = np.array(f['train_encod_data'])
                elif data_split in ["adapt"]:
                    raw_data = np.array(f['valid_encod_data'])
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data[:, None, ::skip_steps, :]
        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)
        t_eval = np.repeat(t_eval[None,:], n_envs, axis=0)

        super().__init__(dataset, t_eval, adaptation, traj_prop_min, use_full_traj)

## Create LinguisticsDataset, inherits from SleepDataset, and does the same thing
class LinguisticsDataset(SleepDataset):
    pass