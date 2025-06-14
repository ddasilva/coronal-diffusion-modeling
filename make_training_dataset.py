import glob

from astropy.io import fits
import numpy as np
import pyshtools
import tqdm
import h5py


def main(root_dir, out_file):
    files = glob.glob(f"{root_dir}/*R000*.fits")
    items = []

    for file in tqdm.tqdm(files, desc="Processing files"):
        for G, H in enumerate_variations(file):
            items.append(
                np.array(
                    [G[np.tril_indices(G.shape[0])], H[np.tril_indices(H.shape[0])]]
                ).flatten()
            )

    items = np.array(items)

    print(items)

    with h5py.File(out_file, "w") as hdf:
        hdf["X"] = items


def enumerate_variations(file_path):
    fits_file = fits.open(file_path)
    sph_data = fits_file[3].data.copy()
    fits_file.close()

    G = sph_data[0, :, :].T
    H = sph_data[1, :, :].T

    yield G, H

    coeffs_array = np.array([G, H])
    coeffs = pyshtools.SHMagCoeffs.from_array(
        coeffs_array, normalization="schmidt", r0=1
    )

    for rot in range(15, 360, 15):
        rot_coeffs = coeffs.copy().rotate(alpha=rot, beta=0, gamma=0, degrees=True)

        yield rot_coeffs.coeffs[0], rot_coeffs.coeffs[1]


if __name__ == "__main__":
    root_dir = "/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
    out_file = "training_dataset.h5"
    main(root_dir, out_file)

    root_dir = "/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
    out_file = "test_dataset.h5"
    main(root_dir, out_file)
