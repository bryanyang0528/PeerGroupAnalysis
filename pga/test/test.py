
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
pg = PeerGroup(df, n_clusters=10, random_state=0, n_init=1)

def test_data_len():
    assert_equal(len(df), 100)

def test_describe():
    desc_array = np.array([[  6. ,  10. ,  95.5],
                          [  1. ,  11. ,  85. ],
                          [  7. ,  12. ,  73.5],
                          [  3. ,  14. ,  60.5],
                          [  0. ,  13. ,  47. ],
                          [  4. ,  11. ,  35. ],
                          [  9. ,   9. ,  25. ],
                          [  2. ,   7. ,  17. ],
                          [  8. ,   6. ,  10.5],
                          [  5. ,   7. ,   4. ]])
    pg_desc = pg.describe()
    assert_array_equal(pg_desc.values, desc_array)

def test_head():
    assert_array_equal(pg.head().values, pg.get_data().head().values)

def test_get_data():
    head_array = np.array([[ 1.,  5.,  1.],
                           [ 1.,  5.,  2.],
                           [ 1.,  5.,  3.],
                           [ 1.,  5.,  4.],
                           [ 1.,  5.,  5.]])

    assert_array_equal(pg.get_data().head().values, head_array)
