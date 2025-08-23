import h5py
import json
import numpy as np
import dask.array  as da
from dask.diagnostics import ProgressBar

import config



def main():
    # Load scalers
    with open(config.scalers_path) as fh:
        scalers = json.load(fh)

    mean = np.array(scalers["mean"])
    std = np.array(scalers["std"])

    # Try to find good starting points within this normalized space
    # for different conditions
    hdf = h5py.File(config.test_dataset_path)
    X = (da.from_array(hdf['X']) - mean) / std
    radio_flux = da.from_array(hdf['radio_fluxes'])

    # uncomment to make code run faster for testing
    #X = X[:1000]
    #radio_flux = radio_flux[:1000]
    
    tasks = [
        (config.seed_helper_max, radio_flux > 0.75),
        (config.seed_helper_min, radio_flux < 0.25),
    ]

    for out_file, mask in tasks:
        # Expand mask to match X's shape for element-wise operations
        mask_expanded = mask[:, None]  # Shape becomes (9538560, 1)

        X_masked = da.where(mask_expanded, X, np.nan)
        mean_helper = da.nanmean(X_masked, axis=0)
        std_helper = da.nanstd(X_masked, axis=0)

        with ProgressBar():
            mean_helper, std_helper = da.compute(mean_helper, std_helper)
        
        out = {
            "mean": mean_helper.tolist(),
            "std": std_helper.tolist(),
        }
        
        with open(out_file, "w") as fh:
            json.dump(out, fh, indent=4)
            print(f"Seed helpers written to {out_file}")

            
if __name__ == "__main__":
    main()

