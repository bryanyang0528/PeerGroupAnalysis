import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import statsmodels.api as sm
from statsmodels.formula.api import ols

class PeerGroup():
    def __init__(self, df, **kwargs):
        self._df_grouped = self.clustering(df, **kwargs)
    
    def clustering(self, df, **kwargs):
        if df.shape[1] >= 2 and isinstance(df, pd.core.frame.DataFrame):
            pass
        else:
            return ValueError("length of column must >= 2")
        ids = df.iloc[:,0].values
        vals = df.iloc[:,1].values.reshape(-1,1)
        model = KMeans(**kwargs)
        model.fit(vals)
        label = model.labels_
        df_g = pd.DataFrame({'id':ids, 'value': vals.reshape(-1,), 'label':label})
        return df_g
    
    def des_label(self):
        if self._df_grouped is not None:
            return self._df_grouped.label.value_counts()
    
    def describe(self, ascending=False):
        if self._df_grouped is not None:
            return self._df_grouped.groupby('label').\
                        agg({'id':len,'value':np.mean}).\
                        sort('value', ascending=ascending).\
                        rename(columns={'value':'mean','id':'count'}).reset_index()
                    
    def get_data(self):
        return self._df_grouped
    
    def head(self):
        return self._df_grouped.head()
    
    def anova(self):
        df = self.get_data()
        model = ols('value ~ label', data = df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table

    def get_within_ss(self):
        df = self.get_data()
        df['value'] = df['value']/(max(df['value'])-min(df['value']))
        return sum((df.groupby('label').std()['value'] ** 2) * (df.groupby('label').count()['value'] -1))
