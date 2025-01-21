import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy as sp
import json
import h5py
import itertools
from multiprocessing import Pool
import pymc as pm
import sys
import time

print('Libraries loaded')

# Day of data
day = sys.argv[1]

# Define the path to save the synthetic data
path = f'./{day}_99_synth'

# Define the frequency range
frequency = np.linspace(1/(1440*60), 1/120, 2000, dtype=np.float32)

# Define the path to the original data
original_data_path = f'./{day}/updated/'

print('Loading periodogram data')
# Read the data from the HDF5 file
with h5py.File(original_data_path+f'{day}_periodograms.h5', mode='r') as f:
    periodograms = f['periodograms'][:]
print(f'Periodogram data loaded, shape: {periodograms.shape}')


# Define the x and y ranges
xrange = range(2048)
yrange = range(2048)
# Generate a list of indices
indices_list = list(itertools.product(xrange, yrange))
# Filter the indices to keep only those within a certain distance from the center
inside_indices_list = list(filter(lambda x: np.sqrt((x[0]-1024)**2 + (x[1]-1024)**2) <= 830, indices_list))
# Randomly select 500 indices from the filtered list
small_region = np.random.randint(0, len(inside_indices_list), 10000)
# Define the number of generated parameters
generated_params = 100

def noise_model(nu, a, alpha, b):
    """
    Calculate the noise model given the frequency and parameters.

    Args:
        nu (array-like): Frequency array.
        a (float): Parameter a.
        alpha (float): Parameter alpha.
        b (float): Parameter b.

    Returns:
        array-like: Noise model.
    """
    return a * nu**(-alpha) + b


def get_real_params(index):
    """
    Get the real distribution parameters for a given pixel.

    Args:
        x_pix (int): X-coordinate of the pixel.
        y_pix (int): Y-coordinate of the pixel.

    Returns:
        tuple: Real distribution parameters.
    """
    print('MCMC starts')
    periodogram = periodograms[index]
    var_names = ['a', 'alpha', 'b']
    basic_model = pm.Model()
    with basic_model:
        # Priors for unknown model parameters
        a = pm.Uniform(var_names[0], 0, 1)
        alpha = pm.Uniform(var_names[1], 0, 10)
        b = pm.Uniform(var_names[2], 0, 1)
        # Model of the spectral density
        Sj = a*frequency**(-alpha) + b
        # Likelihood
        likelihood = pm.Exponential(f"likelihood", scale=Sj, observed=periodogram)
    

    with basic_model:
        trace = pm.sample(draws=4000, tune=500, chains=4, cores=1, progressbar=False)

    basic_samples = trace.posterior.stack(sample=['chain', 'draw'])

    basic_a_samples = np.array(basic_samples['a'])
    basic_alpha_samples = np.array(basic_samples['alpha'])
    basic_b_samples = np.array(basic_samples['b'])



    basic_a_samples = np.array(basic_samples['a'])
    basic_a_vals = np.linspace(basic_a_samples.min(), basic_a_samples.max(), 3000)
    # alpha
    basic_alpha_samples = np.array(basic_samples['alpha'])
    basic_alpha_vals = np.linspace(basic_alpha_samples.min(), basic_alpha_samples.max(), 3000)
    # B
    basic_b_samples = np.array(basic_samples['b'])
    basic_b_vals = np.linspace(basic_b_samples.min(), basic_b_samples.max(), 3000)

    # kernel density estimation of posterior distributions
    # Best fit comes from the mode of the distribution
    kde_a = sp.stats.gaussian_kde(basic_a_samples)
    best_fit_a = basic_a_vals[np.argmax(kde_a(basic_a_vals))]

    kde_alpha = sp.stats.gaussian_kde(basic_alpha_samples)
    best_fit_alpha = basic_alpha_vals[np.argmax(kde_alpha(basic_alpha_vals))]

    kde_b = sp.stats.gaussian_kde(basic_b_samples)
    best_fit_b = basic_b_vals[np.argmax(kde_b(basic_b_vals))]

    # Compute the best fit line
    basic_bf = noise_model(frequency, best_fit_a, best_fit_alpha, best_fit_b) 
    basic_bf = np.array(basic_bf, dtype=np.float32)

    # Using all the samples -> compute sample model lines
    sampled_fits = np.zeros(shape=(len(basic_a_samples), 2000))
    for i, (A, Alpha, B) in enumerate(zip(basic_a_samples, basic_alpha_samples, basic_b_samples)):
        sampled_fits[i, :] = noise_model(frequency, A, Alpha, B)

    # Compute "synthetic" periodogram of the sampled models
    sampled_periodogram = np.zeros_like(sampled_fits)
    for k in range(len(sampled_fits)):
        sampled_periodogram[k] = sampled_fits[k] * sp.stats.chi2.rvs(df=2, size=2000)/2

    # Compute the Rj statistic -> sampled periodogram divided by the best fit -> gives idea of frequency peak height statistics
    Rj_sampled = 2 * sampled_periodogram / basic_bf

    # Compute the confidence line using the sample model lines -> 99% confidence
    def confidence_line(sampled_data, nu, target_significance=1):
        significance = np.zeros(len(nu), dtype=np.float32)
        # for each frequency compute the significance level
        for freq in range(len(nu)):
            # kde estimation of the frequency peak heights
            kde_sampled_data = sp.stats.gaussian_kde(sampled_data[:, freq])
            inf_int_limit = np.percentile(sampled_data[:, freq], 99)
            epsilon = 5e-4
            # We search for inferior integration limit that produces:
            # integral from LowerLimit (what we want) to inf of the kde_sampled_data = 0.01
            # p_val = P(Rj > inf_int_limit) = 0.01
            p_val = kde_sampled_data.integrate_box_1d(inf_int_limit+epsilon, np.inf)
            while p_val >= (target_significance/100):
                # update the lower limit by small increases
                epsilon += 5e-4
                p_val = kde_sampled_data.integrate_box_1d(inf_int_limit+epsilon, np.inf)
            # significance level
            significance[freq] = inf_int_limit + epsilon
        return significance

    # Compute the confidence line
    sig_line = confidence_line(Rj_sampled, frequency)
    conf_line = basic_bf/2 * sig_line

    try:
        popt, _ = sp.optimize.curve_fit(noise_model, frequency, conf_line, p0=[best_fit_a, best_fit_alpha, best_fit_b], maxfev=10000)
        conf_a, conf_alpha, conf_b = popt
    except:
        conf_a, conf_alpha, conf_b = 0, 0, 0

    #return best fit and confidence line parameters
    return best_fit_a, best_fit_alpha, best_fit_b, conf_a, conf_alpha, conf_b

def create_synth_data(args):
    """
    Create synthetic data for a given set of arguments.

    Args:
        args (tuple): Tuple containing the index and random index.

    Returns:
        None
    """
    ind, rand_ind = args
    start_time = time.perf_counter()
    synth_best_fit_params = np.zeros(shape=(generated_params, 3))
    synth_conf_params = np.zeros_like(synth_best_fit_params)
    synthetic_per = np.zeros(shape=(generated_params, 2000))

    best_fit_a, best_fit_alpha, best_fit_b, \
    conf_a, conf_alpha, conf_b = get_real_params(rand_ind)

    for k in range(generated_params):
        synth_best_fit_params[k, :] = best_fit_a, best_fit_alpha, best_fit_b
        synth_conf_params[k, :] = conf_a, conf_alpha, conf_b
        synthetic_per[k, :] = noise_model(frequency, *synth_best_fit_params[k])*sp.stats.chi2.rvs(df=2, size=2000)/2
    
    filename = path + f'/synth_{day}_{ind}.json'
    output = {
                'best_fit_params': synth_best_fit_params.tolist(),
                'conf_params': synth_conf_params.tolist(),
                'periodograms': synthetic_per.tolist()
         }
#a
    with open(filename, 'w') as f:
        json.dump(output, f)
    
    end_time = time.perf_counter()
    print(f'Index {ind} completed in {(end_time-start_time)/60:.2f} minutes')



# Use multiprocessing to create synthetic data for each random index
with Pool(24) as p:
    p.map(create_synth_data, enumerate(small_region))
