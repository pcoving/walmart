import pymc as mc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

def compute_error(y_hat, y, is_holiday):
    w = np.ones(is_holiday.shape)
    w[np.asarray(is_holiday, dtype=np.bool)] = 5.

    return np.sum(np.dot(w, np.abs(y_hat - y)))/np.sum(w)


df = pd.read_csv('train.csv', parse_dates=['Date'])
df['weekofyear'] = df.Date.apply(lambda dt: dt.weekofyear)

encoder = OneHotEncoder()
# start with stores
X = encoder.fit_transform(df.Store.values[:,np.newaxis]).todense()
# add in depts
X = np.hstack([X, encoder.fit_transform(df.Dept.values[:,np.newaxis]).todense()])
# add in week of year
X = np.hstack([X, encoder.fit_transform(df.weekofyear.values[:,np.newaxis]).todense()])
# add in holidays
X = np.hstack([X, np.asarray(df.IsHoliday.values[:,np.newaxis], dtype=np.int)])
# add in bias term
X = np.hstack([X, np.ones([X.shape[0],1])])

X = np.asarray(X)
               
y = df.Weekly_Sales.values

y = np.clip(y, 0, 1e20)

#X, y = shuffle(X, y, random_state=0)

beta = mc.Uninformative('beta', value=[y.mean()]*X.shape[1])
mu_pred = mc.Lambda('mu_pred', lambda beta=beta, X=X: np.dot(X, beta))
y_obs = mc.Poisson('y_obs', mu=mu_pred, value=y, observed=True)

model = mc.Model([beta, mu_pred, y_obs])
mc.MCMC(model).sample(40000, 20000, 50, progress_bar=True)

beta_trace = beta.trace.gettrace()[300:,:]
mu = np.dot(X, beta_trace.T)
y_samples = np.random.poisson(mu)

#y_hat = np.mean(y_samples, axis=1)
y_hat = np.median(y_samples, axis=1)

print compute_error(y_hat, y, df.IsHoliday)
    

