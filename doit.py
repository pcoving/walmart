import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, Imputer
from pandas.core.indexing import IndexingError
import csv
MISSING = -1e6
import pickle

# XXXXX compute ratios? like normalize relative to median?

def compute_error(y_hat, y, is_holiday):
    w = np.ones(is_holiday.shape)
    w[np.asarray(is_holiday, dtype=np.bool)] = 5.

    return np.sum(np.dot(w, np.abs(y_hat - y)))/np.sum(w)

def make_submission(test_id, test_sales):
    with open('submission.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Weekly_Sales'])
        for id, sales in zip(test_id, test_sales):
            writer.writerow([id, sales])

train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])

df = pd.concat([train, test])

df['Id'] = df.apply(lambda row: '_'.join(map(str, [row['Store'], row['Dept'], row['Date'].strftime('%Y-%m-%d')])), axis=1)

meta = pd.read_csv('features.csv', parse_dates=['Date'])
df = pd.merge(df, meta, how='left', on=['Store', 'Date'])
df['IsHoliday'] = df.IsHoliday_x

stores = pd.read_csv('stores.csv')
df = pd.merge(df, stores, how='left', on='Store')

df['Type_A'] = df.Type == 'A'
df['Type_B'] = df.Type == 'B'
df['Type_C'] = df.Type == 'C'

df['dayofyear'] = df.Date.apply(lambda dt: dt.dayofyear)
df['weekofyear'] = df.Date.apply(lambda dt: dt.weekofyear)
df['weekofyear_abs'] = df.Date.apply(lambda dt: dt.weekofyear + (dt.year-2009)*52)
df['year'] = df.Date.apply(lambda dt: dt.year)

print 'computing lagged features...'
for offset, label in zip([52, 52+1, 52-1, 52*2, 52*2+1, 52*2-1, 1, -1, 2, -2],
                         ['lastyear', 'lastyear_m1', 'lastyear_p1', 'lastyear2', 'lastyear2_m1', 'lastyear2_p1', 'm1', 'p1', 'm2', 'p2']):
    #df2 = df.copy()
    df2 = pd.DataFrame({k: df[k] for k in ['Store', 'Dept', 'weekofyear_abs'] + ['Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'dayofyear'] + ['MarkDown' + str(i) for i in range(1,6)]})
    df2['weekofyear_abs'] = df['weekofyear_abs'] + offset
    df = pd.merge(left=df, right=df2, how='left', on=['Store', 'Dept', 'weekofyear_abs'], suffixes=['', '_'+label])
    for key in df.keys():
        if label in key:
            df[key][pd.isnull(df[key])] = MISSING

df['dayofyear_lastyear_diff'] = df['dayofyear'] - df['dayofyear_lastyear']
df['dayofyear_lastyear_diff'][df['dayofyear_lastyear_diff'] > 20] = MISSING
        
for i in range(1,6):
    df['MarkDown'+str(i)][pd.isnull(df['MarkDown'+str(i)])] = MISSING

print 'computing new features...'        
new_features = {}
# there's some leakage here....
# should be number of dept and weekly sales BEFORE the date of the row

new_features['dept_count'] = {'gb': df.groupby('Store').Dept.unique().apply(len), 'left_on': 'Store'}

new_features['sd_min_weekofyear_abs'] = {'gb': df.groupby(['Store', 'Dept']).weekofyear_abs.min(), 'left_on': ['Store', 'Dept']}
new_features['sd_max_weekofyear_abs'] = {'gb': df.groupby(['Store', 'Dept']).weekofyear_abs.max(), 'left_on': ['Store', 'Dept']}

for per in range(0,101,10):
    new_features['store_sales_'+str(per)] = {'gb': df.groupby('Store').Weekly_Sales.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Store'}
    new_features['dept_sales_'+str(per)] = {'gb': df.groupby('Dept').Weekly_Sales.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Dept'}
    #new_features['dept_weekofyear_sales_'+str(per)] = {'gb': df.groupby(['Dept', 'weekofyear']).Weekly_Sales.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Dept'}
    new_features['weekofyear_sales_'+str(per)] = {'gb': df.groupby('weekofyear').Weekly_Sales.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'weekofyear'}
    new_features['store_Temperature_'+str(per)] = {'gb': df.groupby('Store').Temperature.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Store'}
    new_features['store_CPI_'+str(per)] = {'gb': df.groupby('Store').CPI.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Store'}
    new_features['store_Unemployment_'+str(per)] = {'gb': df.groupby('Store').Unemployment.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Store'}
    new_features['store_Fuel_Price_'+str(per)] = {'gb': df.groupby('Store').Fuel_Price.apply(lambda a: np.percentile(a[a == a], per)), 'left_on': 'Store'}
    for i in range(1,6):
        new_features['store_MarkDown'+str(i)+'_'+str(per)] = {'gb': df.groupby('Store')['MarkDown'+str(i)].apply(lambda a: np.percentile(a[a != MISSING], per)), 'left_on': 'Store'}

for per in [0,50,100]:
    new_features['store_dept_sales_'+str(per)] = {'gb': df.groupby(['Dept', 'Store']).Weekly_Sales.apply(lambda a: np.percentile(a[a == a], per) if len(a[a == a]) > 10 else MISSING), 'left_on': ['Dept', 'Store']}
    
for top in range(5):
    print top
    new_features['weekofyear_top'+str(top)] = {'gb': df[~pd.isnull(df.Weekly_Sales)].groupby('Dept').apply(lambda row: list(row['weekofyear'])[np.argsort(list(row['Weekly_Sales']))[::-1][top]]), 'left_on': ['Dept']}
    new_features['weekofyear_bottom'+str(top)] = {'gb': df[~pd.isnull(df.Weekly_Sales)].groupby('Dept').apply(lambda row: list(row['weekofyear'])[np.argsort(list(row['Weekly_Sales']))[top]]), 'left_on': ['Dept']}

new_features['date_dept_count'] = {'gb': df.groupby(['Store', 'weekofyear_abs']).Dept.unique().apply(len), 'left_on': ['Store', 'weekofyear_abs']}

for k, v in new_features.items():
    df = pd.merge(df, pd.DataFrame({k: v['gb']}), how='left', left_on=v['left_on'], right_index=True)

df['sd_min_weekofyear_abs_diff'] = df.weekofyear_abs - df.sd_min_weekofyear_abs
df['sd_max_weekofyear_abs_diff'] = df.sd_max_weekofyear_abs - df.weekofyear_abs

print 'computing holiday features...'

df['Date_str'] = df.Date.apply(lambda d: str(d)[:10])
df['holiday_super_bowl'] = (df.Date_str == '2010-02-12') | (df.Date_str == '2011-02-11') | (df.Date_str == '2012-02-10')  | (df.Date_str == '2013-02-08')
df['holiday_labor_day'] = (df.Date_str == '2010-09-10') | (df.Date_str == '2011-09-09') | (df.Date_str == '2012-09-07')  | (df.Date_str == '2013-09-06')
df['holiday_thanksgiving'] = (df.Date_str == '2010-11-26') | (df.Date_str == '2011-11-25') | (df.Date_str == '2012-11-23')  | (df.Date_str == '2013-11-29')
df['holiday_christmas'] = (df.Date_str == '2010-12-31') | (df.Date_str == '2011-12-30') | (df.Date_str == '2012-12-28')  | (df.Date_str == '2013-12-27')

holidays = {}
holidays['new_year'] = ['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01']
holidays['super_bowl'] = ['2010-02-07', '2011-02-06', '2012-02-05', '2013-02-03']
holidays['valentines'] = ['2010-02-14', '2011-02-14', '2012-02-14', '2013-02-14']
holidays['easter'] = ['2010-04-04', '2011-04-24', '2012-04-08', '2013-03-31']
holidays['mothers_day'] = ['2010-05-09', '2011-05-08', '2012-05-13', '2013-05-12']
holidays['memorial'] = ['2010-05-31', '2011-05-30', '2012-05-28', '2013-05-27']
holidays['fathers_day'] = ['2010-06-20', '2011-06-19', '2012-06-17', '2013-06-16']
holidays['independence'] = ['2010-07-04', '2011-07-04', '2012-07-04', '2013-07-04']
holidays['labor_day'] = ['2010-09-06', '2011-09-05', '2012-09-03', '2013-09-02']
holidays['columbus'] = ['2010-10-11', '2011-10-10', '2012-10-08', '2013-10-14']
holidays['halloween'] = ['2010-10-31', '2011-10-31', '2012-10-31', '2013-10-31']
holidays['thanksgiving'] = ['2010-11-25', '2011-11-24', '2012-11-22', '2013-11-28']
holidays['christmas'] = ['2010-12-25', '2011-12-25', '2012-12-25', '2013-12-25']

holiday_features = []

for hday in holidays:
    holiday_features += ['days_since_'+hday]
    df['days_since_'+hday] = MISSING
    for date in holidays[hday]:
        #tmp = df.Date.apply(lambda d: max(min((d-pd.Timestamp(date)).days, 30), -30))
        tmp = (df.Date - pd.Timestamp(date)).apply(int)/(24*60*60*1e9)
        mask = tmp.abs() < df['days_since_'+hday].abs()
        df['days_since_'+hday][mask] = tmp[mask]

df_val, df_train = df[pd.isnull(df.Weekly_Sales)], df[~pd.isnull(df.Weekly_Sales)]
df_train = df_train[df_train.Date > datetime.date(2011, 02, 10)]

#df_val, df_train = df[(df.Date > datetime.date(2012, 1, 27)) & (~pd.isnull(df.Weekly_Sales))], df[(df.Date <= datetime.date(2012, 1, 27)) & (df.Date > datetime.date(2011, 02, 10))]
#df_val, df_train = df[(df.Date > datetime.date(2012, 1, 27)) & (~pd.isnull(df.Weekly_Sales))], df[(df.Date <= datetime.date(2012, 1, 27))]

#features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'weekofyear', 'lastyear_sales', 'store_sales_avg', 'dept_sales_avg', 'dept_count']

features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'weekofyear', 'weekofyear_abs', 'dayofyear_lastyear_diff']
features += ['Type_A', 'Type_B', 'Type_C', 'Size']
features += ['MarkDown' + str(i) for i in range(1,6)]
features += ['sd_min_weekofyear_abs_diff', 'sd_max_weekofyear_abs_diff']

for feat in ['Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + ['MarkDown' + str(i) for i in range(1,6)]:
    features += [feat + '_lastyear', feat + '_lastyear_m1', feat + '_lastyear_p1']
    features += [feat + '_lastyear2', feat + '_lastyear2_m1', feat + '_lastyear2_p1']

for feat in ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + ['MarkDown' + str(i) for i in range(1,6)]:
    features += [feat + '_m1', feat + '_p1', feat + '_m2', feat + '_p2']
    
features += new_features.keys()

#features += ['holiday_super_bowl', 'holiday_labor_day', 'holiday_thanksgiving', 'holiday_christmas']
features += holiday_features
    
#features += store_names + dept_names

X_train = np.vstack([df_train[feat].values for feat in features]).T
X_val = np.vstack([df_val[feat].values for feat in features]).T
y_train, y_val = df_train.Weekly_Sales.values, df_val.Weekly_Sales.values

imputer = Imputer()
X_val = imputer.fit_transform(X_val)

# oversample for metric (doesnt seem to help)
#y_train = np.hstack([y_train] +  [y_train[X_train[:,0] == 1] for i in range(4)])
#X_train = np.vstack([X_train] +  [X_train[X_train[:,0] == 1] for i in range(4)])

# 6 0.85 1409 1587.43549111 learning_rate = 0.025

# auto 6 0.85 296 1604.76743664

# with days since
# 5 0.85 137 1662.08770224
# 6 0.85 99 1656.39463747
# 7 0.85 195 1640.11291286
# 5 0.9 277 1648.46551374
# 6 0.9 140 1659.25707338
# 7 0.9 129 1656.08793599

# without days since
# 5 0.9 288 1614.88438962
# 6 0.85 147 1622.1854561

# with days since, whole training set
#auto 5 0.85 496 1750.88615433
#auto 6 0.85 496 1716.79292423
#auto 7 0.85 500 1686.50046478
#auto 5 0.9 426 1789.32247197
#auto 6 0.9 418 1740.63649081
#auto 7 0.9 499 1717.13528702
#auto 5 0.95 500 1765.80135795

#auto 7 0.85 573 1682.88618861
#auto 8 0.85 479 1685.79307767

# 0.01 15 auto 7 0.85 55 1714.7518394
# 0.005 41 auto 7 0.85 342 1658.49437386
# 0.001 177 auto 7 0.85 600 1689.1435265
assert False

if True:
    for subsample in [0.85]:
        for max_depth in [7]:
            for max_features in ['auto']:
                clf = GradientBoostingRegressor(n_estimators=600, loss='lad', learning_rate=0.1,
                                                max_depth=max_depth, verbose=1, subsample=subsample,
                                                max_features=max_features, random_state=1)
                clf.fit(X_train, y_train)
            
                errors = np.asarray([compute_error(y_hat, y_val, df_val.IsHoliday.values) for y_hat in clf.staged_predict(X_val)])
                n_est = np.argmin(errors)+1
                print max_features, max_depth, subsample, n_est, min(errors)

assert False

df_train['y_hat'] = clf.predict(X_train)
df_val['y_hat'] = clf.predict(X_val)
post_features = []
for offset, label in zip([-4, -3, -2,-1,0,1,2,3,4], ['p4', 'p3', 'p2', 'p1', '0', 'm1', 'm2', 'm3', 'm4']):
    print offset
    df2_train = pd.DataFrame({k: df_train[k] for k in ['Store', 'Dept', 'weekofyear_abs', 'y_hat'] })
    df2_train['weekofyear_abs'] = df_train['weekofyear_abs'] + offset
    if 'y_hat_'+label in df_train.keys():
        del df_train['y_hat_'+label]
    df_train = pd.merge(left=df_train, right=df2_train, how='left', on=['Store', 'Dept', 'weekofyear_abs'], suffixes=['', '_'+label])
    df2_val = pd.DataFrame({k: df_val[k] for k in ['Store', 'Dept', 'weekofyear_abs', 'y_hat'] })
    df2_val['weekofyear_abs'] = df_val['weekofyear_abs'] + offset
    if 'y_hat_'+label in df_val.keys():
        del df_val['y_hat_'+label]
    df_val = pd.merge(left=df_val, right=df2_val, how='left', on=['Store', 'Dept', 'weekofyear_abs'], suffixes=['', '_'+label])
    post_features += ['y_hat_'+label]

    
# from 1686.471066 to ...
# 7 99 1674.98835314
X_train2 = np.vstack([df_train[feat].values for feat in post_features]).T
y_train2 = y_train.copy()
X_val2 = np.vstack([df_val[feat].values for feat in post_features]).T
X_train2[X_train2 != X_train2] = MISSING
X_val2[X_val2 != X_val2] = MISSING
for max_depth in [7]:
    clf2 = GradientBoostingRegressor(n_estimators=600, loss='lad', learning_rate=0.05,
                                    max_depth=max_depth, verbose=1, subsample=0.9,
                                    max_features='auto', random_state=1)
    clf2.fit(X_train2, y_train2)
    
    errors = np.asarray([compute_error(y_hat, y_val, df_val.IsHoliday.values) for y_hat in clf2.staged_predict(X_val2)])
    n_est = np.argmin(errors)+1
    print max_depth, n_est, min(errors)

# features = [tup[0] for tup in sorted_importance if tup[1] > 0.00296]
# 91 features
'''
clf = GradientBoostingRegressor(n_estimators=3000, loss='lad', learning_rate=0.05, max_depth=7, subsample=0.85, verbose=2, random_state=1)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_val)
make_submission(df_val.Id, y_hat)
'''
assert False

encoder = OneHotEncoder()
dept = encoder.fit_transform(df.Dept.values[:, np.newaxis]).todense()
dept_names = []
for idx, feat in enumerate(encoder.active_features_):
    df['dept_' + str(feat)] = dept[:,idx]
    dept_names.append('dept_' + str(feat))
store = encoder.fit_transform(df.Dept.values[:, np.newaxis]).todense()
store_names = []
for idx, feat in enumerate(encoder.active_features_):    
    df['store_' + str(feat)] = store[:,idx]
    store_names.append('store_' + str(feat))

for i in range(1,6):
    train['MarkDown'+str(i)][pd.isnull(train['MarkDown'+str(i)])] = -1e6
    val['MarkDown'+str(i)][pd.isnull(val['MarkDown'+str(i)])] = -1e6

test = pd.read_csv('test.csv', parse_dates=['Date'])
for frame in [train, test, val]:
    frame['dayofyear'] = frame.Date.apply(lambda dt: dt.dayofyear)

'''        
a = train.groupby(by=['Store', 'Dept'])
b = pd.DataFrame({'store_dept_dayofyear': a.dayofyear.apply(lambda lt: np.asarray(list(lt) + [400])), 'store_dept_sales': a.Weekly_Sales.apply(lambda lt: np.asarray(list(lt) + [0.0]))})
train = pd.merge(train, b, how='left', left_on=('Store', 'Dept'), right_index=True)
train['dayofyear_sales'] = train.apply(lambda row: row['store_dept_sales'][np.argmin(np.abs(row['store_dept_dayofyear'][row['store_dept_dayofyear'] != row['dayofyear']]-row['dayofyear']))], axis=1)

test = pd.merge(test, b, how='left', left_on=('Store', 'Dept'), right_index=True)
test['dayofyear_sales'] = test.apply(lambda row: row['store_dept_sales'][np.argmin(np.abs(row['store_dept_dayofyear'][row['store_dept_dayofyear'] != row['dayofyear']]-row['dayofyear']))], axis=1)
compute_error(train.dayofyear_sales, train.Weekly_Sales.values, train.IsHoliday.values)
'''
    
'''
for store in range(1,46):
    train_dept = train[train.Store == store].Dept.unique()
    test_dept = test[test.Store == store].Dept.unique()
    for dept in test_dept:
        if dept not in train_dept:
            print store, dept            
'''
            

features = ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'dayofyear']
features += store_names + dept_names
#features += new_features.keys()
#features += ['MarkDown' + str(i) for i in range(1,6)]

X_train = np.vstack([train[feat].values for feat in features]).T

X_val = np.vstack([val[feat].values for feat in features]).T

y_train, y_val = train.Weekly_Sales.values, val.Weekly_Sales.values

# oversample for metric
#y_train = np.hstack([y_train] +  [y_train[X_train[:,0] == 1] for i in range(4)])
#X_train = np.vstack([X_train] +  [X_train[X_train[:,0] == 1] for i in range(4)])

'''
2 995 8413.32726137
4 996 8132.31507432
6 541 8044.28211036
8 369 7964.68077228
10 326 7974.26743807
'''

for max_depth in range(2,22,3):
    clf = GradientBoostingRegressor(n_estimators=1000, loss='lad', max_depth=max_depth)
    clf.fit(X_train, y_train)

    errors = np.asarray([compute_error(y_hat, y_val, val.IsHoliday.values) for y_hat in clf.staged_predict(X_val)])
    n_est = np.argmin(errors)+1
    print max_depth, n_est, min(errors)

'''
pred = clf.staged_predict(X_val)
for iest, y_hat in enumerate(pred):
    print iest, compute_error(y_hat, y_val, val.IsHoliday.values)
'''
assert False
    
'''    
meta = pd.read_csv('features.csv', parse_dates=['Date'])
df = pd.merge(train, meta, how='left', on=['Store', 'Date'])
df['IsHoliday'] = df.IsHoliday_x
features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'dayofyear']
'''

features = ['Store', 'Dept', 'IsHoliday', 'dayofyear']

X_train = np.vstack([train[feat].values for feat in features]).T
X_val = np.vstack([val[feat].values for feat in features]).T
y_train, y_val = train.Weekly_Sales.values, val.Weekly_Sales.values

y_hat = np.median(y_train)*np.ones(y_val.shape)
for idx in range(len(y_hat)):
    x = X_val[idx]
    tmp = train[(train.Store == x[0]) & (train.Dept == x[1]) & (train.IsHoliday == x[2])]
    match = tmp[abs(tmp.dayofyear - x[3]) <= 3]
    if len(match) > 0:
        y_hat[idx] = np.mean(match.Weekly_Sales)
    else:
        print 'no match'
        y_hat[idx] = np.median(y_train)
    if idx%1000 == 0:
        print idx, float(idx)/len(y_hat)
        
    
print compute_error(y_hat, y_val, val.IsHoliday.values)


