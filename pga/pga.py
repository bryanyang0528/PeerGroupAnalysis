import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class PeerGroup():
    def __init__(self, df, **kwargs):
        _df_grouped = self.clustering(df, **kwargs)
    
    def clustering(self, df, **kwargs):
        if df.shape[1] > 2 and isinstance(df, pd.core.frame.DataFrame):
            pass
        else:
            return ValueError("length of column must > 2")
        ids = df.iloc[:,0].values
        vals = df.iloc[:,1].values.reshape(-1,1)
        model = KMeans(**kwargs)
        model.fit(vals)
        label = model.labels_
        df_g = pd.DataFrame({'userid':ids, 'value': vals.reshape(-1,), 'label':label})
        self._df_grouped = df_g
        return df_g
    
    def des_label(self):
        if self._df_grouped is not None:
            return self._df_grouped.label.value_counts()
    
    def describe(self, ascending=False):
        if self._df_grouped is not None:
            return self._df_grouped.groupby('label').\
                        agg({'userid':len,'value':np.mean}).\
                        sort('value', ascending=ascending)
    
    def get_data(self):
        return _df_grouped
