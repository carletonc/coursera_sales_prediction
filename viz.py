import numpy as np
import pandas as pd



def correlation(df, precision=4):
    corr = df.corr()
    corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    return corr.style.background_gradient(cmap='coolwarm').set_precision(precision)