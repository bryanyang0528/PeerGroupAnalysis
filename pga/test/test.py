
import unittest
import pandas as pd
import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import if_safe_multiprocessing_with_blas
from sklearn.utils.testing import assert_raise_message

from pga.pga import PeerGroup

df = pd.read_csv('data/sample_data.csv', sep = ',', header=0)
df_pg = df[['id', 'value']]
pg = PeerGroup(df_pg, n_clusters=10, random_state=0, n_init=1)

def test_data_len():
    assert_equal(len(df), 100)

def test_describe():
    desc_array = np.array([[   8.        ,    7.        ,  964.57142857],
                           [   0.        ,    8.        ,  872.625     ],
                           [   2.        ,   11.        ,  762.90909091],
                           [   9.        ,   13.        ,  683.30769231],
                           [   5.        ,    8.        ,  610.5       ],
                           [   4.        ,    7.        ,  508.71428571],
                           [   6.        ,   12.        ,  402.91666667],
                           [   1.        ,   11.        ,  291.54545455],
                           [   7.        ,   13.        ,  149.92307692],
                           [   3.        ,   10.        ,   54.6       ]])
    pg_desc = pg.describe()
    assert_array_almost_equal(pg_desc.values, desc_array)

def test_head():
    assert_array_equal(pg.head().values, pg.get_data().head().values)

def test_get_data():
    head_array = np.array([[   1.,    7.,  120.],
                           [   2.,    6.,  362.],
                           [   3.,    7.,  185.],
                           [   4.,    9.,  701.],
                           [   5.,    8.,  940.]])

    assert_array_almost_equal(pg.get_data().head().values, head_array)

def test_anova():
    assert_almost_equal(pg.anova()['PR(>F)'][0], 0.68092934459725607)

def test_within_ss():
    assert_almost_equal(pg.get_within_ss(), 0.062407716931187691)

def test_within_ss_decreasing():
    x = []
    for i in range(2, len(df_pg)):
        pg = PeerGroup(df_pg, n_clusters=i, random_state=0, n_init=1)
        x.append(pg.get_within_ss())
    print(x)
    for i in range(1, len(x)):
        print(x[i])
        if x[i] > x[i-1]:
            raise ValueError("within_ss is not decreasing")
