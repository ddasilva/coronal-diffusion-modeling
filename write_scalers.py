import h5py
import json
import numpy as np
import dask.array  as da

def main():
    hdf = h5py.File("test_dataset.h5")
    X = da.from_array(hdf['X'])
    
    mean = X.mean(axis=0).compute()
    std = X.std(axis=0).compute()
    mean_abs = np.abs(X).mean(axis=0).compute()
    mean_square = np.square(X).mean(axis=0).compute()
    
    out = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mean_abs": mean_abs.tolist(),
        "mean_square": mean_square.tolist()
    }

    with open("scalers.json", "w") as fh:
        json.dump(out, fh, indent=4)


if __name__ == "__main__":
    main()
    print("Scalers written to scalers.json")
