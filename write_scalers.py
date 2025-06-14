import h5py
import json


def main():
    hdf = h5py.File("training_dataset.h5")
    X = hdf["X"][:]
    hdf.close()

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    out = {"mean": mean.tolist(), "std": std.tolist()}

    with open("scalers.json", "w") as fh:
        json.dump(out, fh, indent=4)


if __name__ == "__main__":
    main()
    print("Scalers written to scalers.json")
