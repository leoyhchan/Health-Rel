# %%
from scipy.stats import ttest_rel
import pandas as pd

# %%
METRICS = ['accuracies', 'f1_micro', 'f1_rel', 'f1_unrel']

# %%
z = pd.read_csv('test-z.csv')
sw = pd.read_csv('test-stopwords.csv')

# %%
print('H0: Metrics with standardisation = Metrics without standardisation')
print('HA: Metrics with standardisation > Metrics without standardisation')
for m in METRICS:
    a = z[z['standard']][m]
    b = z[~z['standard']][m]
    p = ttest_rel(a, b, alternative='greater').pvalue
    print(f'Metric: {m}, p-value: {p}')

# %%
print('H0: Metrics without stopword removal = Metrics with stopword removal')
print('HA: Metrics without stopword removal > Metrics with stopword removal')
for m in METRICS:
    a = sw[sw['features'] == 'allKeep'][m]
    b = sw[sw['features'] == 'allRem'][m]
    p = ttest_rel(a, b, alternative='greater').pvalue
    print(f'Metric: {m}, p-value: {p:0.4f}')