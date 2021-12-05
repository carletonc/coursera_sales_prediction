import numpy as np
import pandas as pd
#import plotly.graph_objects as go



def correlation(df, precision=4):
    corr = df.corr()
    corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    #data=go.Heatmap(corr)
    #fig = go.Figure(data)
    #fig.show()
    return corr.style.background_gradient(cmap='coolwarm').set_precision(precision)