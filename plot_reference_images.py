from astropy.io import fits

import coronal_diffusion.visualization_tools as vt

import config



def main(wsa_path, image_path):
    G, H = load_coeffs(wsa_path)

    vis = vt.SHVisualizer(G, H, normalization='schmidt')
    fig = vis.visualize_field_lines(r=1.1, grid_density=40)
    fig.write_image(image_path, width=800, height=600)   

    print(f'Wrote {image_path}')


def load_coeffs(wsa_path):
    fits_file = fits.open(wsa_path)
    G, H = fits_file[3].data.copy()
    fits_file.close()

    G = G.T  # written by fortran
    H = H.T
    
    return G, H

    
if __name__ == "__main__":
    main(
        config.ref_min_path,
        'plots/reference_min.png'
    )

    main(
        config.ref_max_path,
        'plots/reference_max.png'
    )
