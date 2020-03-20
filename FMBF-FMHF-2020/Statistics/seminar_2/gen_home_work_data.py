from itertools import chain
from typing import Union

import numpy as np
import numpy.random as rnd
import pandas as pd

rnd.seed(999)

N_HEALTHY = 327
N_ILL = 48

antigen_types = (
    ('H1', 'H2', 'H3', 'H4', 'H5'),
    ('N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7')
)
N_Hs = len(antigen_types[0])
N_Ns = len(antigen_types[1])
true_type = ('H2', 'N4')

affinity = (
    (.57, 1, .31, .75, .03),
    (.03, .05, .02, 1, .01, .1, .78)
)
affinity_numpy = np.fromiter(chain.from_iterable(affinity), dtype=float)

H_mu = 16
H_sigma = 4

N_mu = 48
N_sigma = 21 ** 0.5

NOISE_mu = 0
NOISE_sigma = 0.4

patient_ID_range = pd.Series(
    index=pd.Index(
        range(1, N_HEALTHY + N_ILL + 1),
        name='Patient ID'
    ),
    data=pd.Categorical(
        rnd.permutation(['Ill'] * N_ILL + ['Healthy'] * N_HEALTHY),
        categories=('Healthy', 'Ill'),
    ),
    name='Status'
)

colnames = tuple(
    map(lambda ag_type: f'{ag_type} Ab fluo Intensity', chain.from_iterable(antigen_types))
)

ill_df = pd.DataFrame(
    affinity_numpy * np.hstack(
        (
            rnd.normal(H_mu, H_sigma, size=(N_ILL, N_Hs)),
            rnd.normal(N_mu, N_sigma, size=(N_ILL, N_Ns))
        )
    ),
    columns=colnames,
    index=patient_ID_range[patient_ID_range == 'Ill'].index
)


def squeeze_column(df: pd.DataFrame, col: str, factor: Union[int, float]) -> type(None):
    df[col] += (df[col].mean() - df[col]) * factor


H2_SCALING_FACTOR = .85
N4_SCALING_FACTOR = .75

squeeze_column(ill_df, 'H2 Ab fluo Intensity', H2_SCALING_FACTOR)
squeeze_column(ill_df, 'N4 Ab fluo Intensity', N4_SCALING_FACTOR)

ill_df += rnd.normal(NOISE_mu, NOISE_sigma, size=ill_df.shape)

healthy_df = pd.DataFrame(
    rnd.normal(NOISE_mu, NOISE_sigma, size=(N_HEALTHY, len(colnames))),
    columns=colnames,
    index=patient_ID_range[patient_ID_range == 'Healthy'].index
)

pd.concat((healthy_df, ill_df)).sort_index().to_csv('fluo.tsv', sep='\t')
patient_ID_range.to_csv('patient_status.csv')
