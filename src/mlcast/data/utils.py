import numpy as np
from queue import Queue
from numpy.typing import NDArray
import zarr

def _dim_nan_count(
    mask: NDArray[np.int16], dim: int, delta: int, dim_len: int
) -> NDArray[np.int32]:
    """Compute the number of NaN in each window along a dimension.

    Args:
        mask: Binary array indicating NaN positions.
        dim: Dimension along which to compute.
        delta: Window size along dimension.
        dim_len: Length of the dimension.

    Returns:
        Array of NaN counts for each window position.
    """
    cumsum = np.cumsum(mask, axis=dim, dtype=np.int32)

    # Pad with zeros at the start along 'dim'
    pad_width = [(1, 0) if i == dim else (0, 0) for i in range(3)]
    padded_cumsum = np.pad(cumsum, pad_width=pad_width, mode="constant", constant_values=0)

    # Rolling window difference
    slices_start = [slice(dim_len - delta) if i == dim else slice(None) for i in range(3)]
    slices_end = [slice(delta, dim_len) if i == dim else slice(None) for i in range(3)]

    return padded_cumsum[tuple(slices_end)] - padded_cumsum[tuple(slices_start)]


def _datacube_nan_count(
    chunk: NDArray, deltas: tuple[int, int, int], dim_lengths: tuple[int, int, int]
) -> NDArray[np.int32]:
    """Compute the number of NaN in each datacube within a chunk.

    Args:
        chunk: Data chunk of shape (T, X, Y).
        deltas: Window sizes (Dt, w, h).
        dim_lengths: Chunk dimensions (T, X, Y).

    Returns:
        Array of NaN counts for each possible datacube position.
    """
    nan_mask = np.isnan(chunk).astype(np.int16)

    # Number of NaN along time
    nans_t = _dim_nan_count(nan_mask, dim=0, delta=deltas[0], dim_len=dim_lengths[0])

    # Number of NaN along X x T
    nans_xt = _dim_nan_count(nans_t, dim=1, delta=deltas[1], dim_len=dim_lengths[1])

    # Number of NaN in the datacube (Y x X x T)
    return _dim_nan_count(nans_xt, dim=2, delta=deltas[2], dim_len=dim_lengths[2])



def _process_chunk(
    time_range: tuple[int, int],
    t_start_idx: int,
    data: zarr.Array,
    max_nan: int,
    deltas: tuple[int, int, int],
    steps: tuple[int, int, int],
    valid_starts_gap: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Process a single time chunk and return valid datacube indices.

    Args:
        time_range: Start and end indices of the chunk.
        t_start_idx: Index offset corresponding to start_date.
        data: Zarr array.
        max_nan: Maximum number of NaN in each datacube.
        deltas: Datacube dimensions (Dt, w, h).
        steps: Step sizes (step_T, step_X, step_Y).
        valid_starts_gap: Valid starting time indices without gaps.

    Returns:
        Tuple of (t_indices, x_indices, y_indices).
    """
    start_t, end_t = time_range

    # Load chunk from Zarr (T, X, Y)
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :]
    dim_lengths = chunk.shape

    # Compute NaN counts
    nans_cube_chunk = _datacube_nan_count(chunk, deltas, dim_lengths)
    del chunk

    # Apply threshold mask
    valid_mask = nans_cube_chunk <= max_nan
    del nans_cube_chunk

    # Get indices (relative to chunk)
    idx_t_rel, idx_x, idx_y = np.where(valid_mask)
    del valid_mask

    # Cast to int32
    idx_t_rel = idx_t_rel.astype(np.int32)
    idx_x = idx_x.astype(np.int32)
    idx_y = idx_y.astype(np.int32)

    # Convert relative time indices
    idx_t = idx_t_rel + start_t

    # Keep only time indices in valid_starts_gap
    time_mask = np.isin(idx_t, valid_starts_gap)
    idx_t = idx_t[time_mask] + t_start_idx  # convert to absolute index
    idx_x = idx_x[time_mask]
    idx_y = idx_y[time_mask]

    # Filter by step size
    stride_mask = (idx_t % steps[0] == 0) & (idx_x % steps[1] == 0) & (idx_y % steps[2] == 0)
    idx_t = idx_t[stride_mask]
    idx_x = idx_x[stride_mask]
    idx_y = idx_y[stride_mask]

    return idx_t, idx_x, idx_y

def _file_writer(output_queue: Queue, filename: str, batch_size: int = 1000) -> None:
    """Write results to file from queue in a dedicated thread."""
    with open(filename, "w") as f:
        f.write("t,x,y\n")
        batch = []

        while True:
            item = output_queue.get()

            if item is None:  # Sentinel to stop
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                break

            batch.extend(zip(*item))

            if len(batch) >= batch_size:
                for t, x, y in batch:
                    f.write(f"{t},{x},{y}\n")
                f.flush()
                batch = []

   
