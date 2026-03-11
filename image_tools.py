def generate_psf(shape, sigma):
    """Generate a synthetic Gaussian PSF."""
    psf = np.zeros(shape)
    center = tuple(s // 2 for s in shape)
    psf[center] = 1
    psf = gaussian_filter(psf, sigma=sigma)
    psf /= psf.sum()
    return psf

def deconvolve_channel(channel_img, sigma=2, iterations=15):
    psf = generate_psf(channel_img.shape, sigma)
    return richardson_lucy(channel_img, psf, num_iter=iterations)
    
def normalize_to_8bit(channel_data):
    return ((channel_data - channel_data.min()) / (channel_data.ptp() + 1e-8) * 255).astype(np.uint8)
