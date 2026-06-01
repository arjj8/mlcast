from pipeline_core import Pipeline
import xarray as xr
import fiddle as fdl

cfg = Pipeline()
#cfg.steps.pop(1)
cfg.steps[0].pars.start_date = '2021-07-01'
cfg.steps[0].pars.end_date = '2021-07-31'
cfg.steps[0].pars.w = 256
cfg.steps[0].pars.h = 256
cfg.steps[0].pars.max_nan = 100000
cfg.steps[0].pars.time_depth = 24
cfg.steps[1].pars.conversion = False
pipeline = fdl.build(cfg)

zarr_path = '/dmidata/projects/radar/products/composite/zarrComposite/products/DMI_500m_10min_v011.zarr'
pipeline(zarr_path)
