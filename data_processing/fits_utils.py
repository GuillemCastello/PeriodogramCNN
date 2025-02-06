from astropy.io import fits

def open_fits_fz_file(file):
    with fits.open(file, mode='readonly') as hdul:
        header = hdul[1].header
        data = hdul[1].data
        return header, data
    
def open_fits_file(file):
    with fits.open(file, mode='readonly') as hdul:
        header = hdul[0].header
        data = hdul[0].data
        return header, data

def save_fits_file(path, header, data):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=True)
