import pandas as pd
import numpy as np

data = pd.read_excel('../Data/IBM.xlsx')

last_5_years = data.tail(60)  # 每年 12 个月，5 年共 60 个月
last_20_years = data.tail(240)  # 每年 12 个月，20 年共 240 个月


def bayesian_update(prior_mean, prior_std, observed_data):
    T = len(observed_data)
    sample_mean = np.mean(observed_data)
    sample_var = np.var(observed_data)
    sample_std = np.sqrt(sample_var)

    w = (sample_std ** 2 / T) / (prior_std ** 2 + (sample_std ** 2 / T))

    posterior_mean = w * prior_mean + (1 - w) * sample_mean

    posterior_std = np.sqrt(1 / (1 / prior_std ** 2 + T / sample_std ** 2))

    return posterior_mean, posterior_std

prior_mean = 0.01
prior_stds = [0.005, 0.01, 0.02, 0.04]

observed_5_years = last_5_years["Return"]

observed_20_years = last_20_years["Return"]

print("Results for Last 5 Years Data:")
for prior_std in prior_stds:
    posterior_mean, posterior_std = bayesian_update(prior_mean, prior_std, observed_5_years)
    print(f"Prior Std: {prior_std:.4f}, Posterior Mean: {posterior_mean:.4f}, Posterior Std: {posterior_std:.4f}")

print("\nResults for Last 20 Years Data:")
posterior_mean_20, posterior_std_20 = bayesian_update(prior_mean, 0.04, observed_20_years)
print(f"Prior Std: 0.04, Posterior Mean: {posterior_mean_20:.4f}, Posterior Std: {posterior_std_20:.4f}")
