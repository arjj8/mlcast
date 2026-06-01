import mlcast_datasets
import xarray as xr
import zarr
import dask
import fiddle as fdl 
from fiddle.experimental import auto_config
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
#from loguru import logger
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from utils import _process_chunk,_file_writer
from numpy.typing import NDArray
from tqdm import tqdm
from pathlib import Path
class TilingSampler:
    def __init__(self, pars):
        self.Dt = pars.Dt
        #self.tile_size = pars.tile_size
        self.w = pars.w
        self.h = pars.h
        self.step_T = pars.step_T
        self.step_X = pars.step_X
        self.step_Y = pars.step_Y
        self.max_nan = pars.max_nan
        self.n_workers = pars.n_workers
        self.time_step_minutes = pars.time_step_minutes
        self.data_var = pars.data_var
        self.time_var = pars.time_var
        #self.time_chunk_size = pars.time_chunk_size
        self.start_date = pars.start_date
        self.end_date = pars.end_date
        self.time_depth = pars.time_depth
        self.data_path = pars.data_path

    def __call__(self,zarr_path,dir_paths):#da: xr.DataArray):
        """
        Compute tile indices and build tiles from source dataset and write to:
        csv_path = f"{tiling_id}.samples.csv", where
        tiling_id = f"{da.attrs['dataset_id']}.{self.tile_size}.{self.n_time_window}".
        da shape: [n_time, x, y]
        tiled output shape: [tile_id, n_time_window, x_tile, y_tile]

        The script will be based in the work of Gabriele Franch and Martin Frølund
        """
        self.time_chunk_size = self.Dt * 3
        if self.data_path == None:
            zg = zarr.open(zarr_path, mode="r")
        else:
            zg = zarr.open(self.data_path, mode="r")
        #data = da[self.data_var].values
        data = zg[self.data_var]

        da = xr.open_zarr(zarr_path)

        time_array_full = pd.DatetimeIndex(da[self.time_var].values)
        start_date = pd.to_datetime(self.start_date) if self.start_date else time_array_full[0]
        end_date = pd.to_datetime(self.end_date) if self.end_date else time_array_full[-1]

        mask = (time_array_full >= start_date) & (time_array_full <= end_date)
        valid_indices = np.where(mask)[0]

        t_start_idx = valid_indices[0]
        t_end_idx = valid_indices[-1] + 1

        size_T = t_end_idx - t_start_idx
        size_X = data.shape[1]
        size_Y = data.shape[2]
        time_array = time_array_full[t_start_idx:t_end_idx]

        max_t = size_T - self.Dt + 1

        expected_step = pd.Timedelta(minutes=self.time_step_minutes)
        time_diffs = time_array[1:] - time_array[:-1]
        gaps = (time_diffs != expected_step).astype(int)
        
        window_sum = np.convolve(gaps, np.ones(self.Dt - 1, dtype=int), mode="valid")
        valid_starts_gap = np.where(window_sum == 0)[0]

        estimated_chunk_memory_gb = (self.time_chunk_size * size_X * size_Y * 4) / (1024**3)

         # Prepare time chunks
        t_starts = np.arange(0, max_t, self.time_chunk_size)
        t_ends = np.minimum(t_starts + self.time_chunk_size + self.Dt - 1, size_T)
        t_pairs = np.stack((t_starts, t_ends), axis=1)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")


        output_file = f"valid_datacubes_{start_str}-{end_str}_{self.Dt}x{self.w}x{self.h}_{self.step_T}x{self.step_X}x{self.step_Y}_{self.max_nan}.csv"
        dir_path['cv_filter_nan'] = output_file

        # Create partial function
        process_chunk_partial = partial(
            _process_chunk,
            t_start_idx=t_start_idx,
            data=data,
            max_nan=self.max_nan,
            deltas=(self.Dt, self.w, self.h),
            steps=(self.step_T, self.step_X, self.step_Y),
            valid_starts_gap=valid_starts_gap,
        )  

        output_queue: Queue = Queue(maxsize=100)
        writer_thread = Thread(target=_file_writer, args=(output_queue, output_file, 1000))
        writer_thread.daemon = False
        writer_thread.start()

        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            for hits in tqdm(
                    pool.imap(process_chunk_partial, t_pairs, chunksize=1),
                    total=len(t_starts),
                    desc="Processing time chunks",
        ):
                output_queue.put(hits)

        # Signal writer thread to stop
        output_queue.put(None)
        writer_thread.join()



        raise NotImplementedError


class BinNormSampler:
    def __init__(self, pars):
        self.a_par = pars.a_par
        self.b_par = pars.b_par
        self.conversion = pars.conversion
        self.mean_weight = pars.mean_weight

    def __call__(self, zarr_path,dir_paths):
        """
        Compute sampling/binning indices and write to:
        csv_path = f"{sampling_id}.samples.csv", where
        sampling_id = f"{tiled.attrs['tiling_id']}.{self.aggregation_method}.{self.n_scalar_bins}".
        tiled shape: [tile_id, n_time_window, x_tile, y_tile]
        Also build sampled output:
        output shape [sampled_tile_id, n_time_sample, x_tile, y_tile]
        The script will be based in the work of Gabriele Franch
        """
        zg = zarr.open(zarr_path, mode="r")
        cv = dir_paths['cv_filter_nan']

        raise NotImplementedError

@dataclass
class Tile_pars():
    Dt: int =  24
    w: int =256
    h: int = 256
    step_T: int = 3
    step_X: int = 16
    step_Y: int = 16
    max_nan: int = 10000, 
    n_workers: int = 8
    time_step_minutes: int = 10
    data_var: str = 'dbz'
    time_var: str = 'time'
    #time_chunk_size: int = None
    start_date: str = None
    end_date: str = None
    time_depth: int = 24
    data_path: str = None
    #def __post_init__(self):
     #   if self.time_chunk_size is None:
      #      self.time_chunk_size = self.Dt * 3


@dataclass
class Bin_pars(Tile_pars):
    q_min: float = 1e-4
    scale: int = 1
    mean_weight: float = 0.1
    n_samples: int = 1
    data_path: str = None
    conversion: bool = True
    a_par: float = 0.6
    b_par: float = 2


class DataPipeline:
    def __init__(self,steps):
        self.steps = steps

    def __call__(self, data):
        paths = {}
        for step in self.steps:
            result = step(data,dir_paths=paths)
        return result

@auto_config.auto_config
def build_pipeline(new_steps = []):
    init_steps = [TilingSampler(pars=Tile_pars),
                  BinNormSampler(pars=Bin_pars)]
    init_steps.extend(new_steps)
    pipeline = DataPipeline(init_steps)
    return pipeline

def Pipeline(new_steps = None):
    if new_steps:
        configured_steps  = [
                fdl.Config(stp_class,pars=params) 
                for stp_class,params in new_steps
                ]
    else:
        configured_steps = []

    return build_pipeline.as_buildable(configured_steps)


