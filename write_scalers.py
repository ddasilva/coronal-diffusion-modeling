import h5py
import json
import numpy as np
import dask.array  as da

import config


def main():
    hdf = h5py.File(config.test_dataset_path)
    X = da.from_array(hdf['X'])
    
    mean = X.mean(axis=0).compute()
    std = X.std(axis=0).compute()
    
    out = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    with open(config.scalers_path, "w") as fh:
        json.dump(out, fh, indent=4)


if __name__ == "__main__":
    main()
    print(f"Scalers written to {config.scalers_path}")
