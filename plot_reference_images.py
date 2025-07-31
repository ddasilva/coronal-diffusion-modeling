from astropy.io import fits
import visualization_tools as vt


def load_coeffs(wsa_path):
    fits_file = fits.open(wsa_path)
    G, H = fits_file[3].data.copy()
    fits_file.close()

    G = G.T  # written by fortran
    H = H.T
    
    return G, H


def main(wsa_path, image_path):
    G, H = load_coeffs(wsa_path)

    vis = vt.SHVisualizer(G, H, normalization='schmidt')
    fig = vis.visualize_field_lines(r=1.1, grid_density=40)
    fig.write_image(image_path, width=800, height=600)   

    print(f'Wrote {image_path}')


if __name__ == "__main__":

    #main(
    #    '/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test/wsa_201010010800R000_ahmi.fits',
    #    'reference_min.png'
    #)

    main(
        '/home/ubuntu/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test/wsa_201410110800R001_ahmi.fits',
        'reference_max.png'
    )