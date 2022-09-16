### Python Statistics Fundamentals: How to Describe Your Data ###
### link: https://realpython.com/python-statistics ###

### Calculating Descriptive Statistics ###
# import needed packages #
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

# create some data #
x = [8.0, 1, 2.5, 4, 28.0]
# create some data with missing value - float('nan), math.nan, np.nan
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
x_with_nan

# use all of these functions interchangeably for creating missing value #
# math.isnan(np.nan), np.isnan(math.nan)
# math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

#  create np.ndarray and pd.Series objects that correspond to x and x_with_nan #
y, y_with_nan = np.array(x), np.array(x_with_nan)
y
y_with_nan
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z
z_with_nan

# Measures of Central Tendency #
# calculate mean of five data points in x #
mean_ = sum(x) / len(x)
mean_ # mean = 8.7
# or use built-in python statistics function #
mean_ = statistics.mean(x)
mean_
mean_ = statistics.fmean(x) # faster alternative to mean().
mean_
# calculate mean with missing - still run but return 'nan' #
mean_ = statistics.mean(x_with_nan)
mean_
mean_ = statistics.fmean(x_with_nan)
mean_
# calculate mean use Numpy #
mean_ = np.mean(y)
mean_
#  mean() is a function, but you can use the corresponding method .mean() as well #
mean_ = y.mean()
mean_
# calculate mean with missing using Numpy - still run but return 'nan' #
np.mean(y_with_nan)
y_with_nan.mean()
# calculate mean with missing using Numpy - ignore missing to avoid 'nan' #
np.nanmean(y_with_nan) # mean = 8.7
# calculate mean in pd.Series objects #
mean_ = z.mean()
mean_
z_with_nan.mean() # pandas ignores 'nan' by default

# weighted mean #
0.2 * 2 + 0.5 * 4 + 0.3 * 8 # weighted mean = 4.8
# weighted mean in pure Python # 
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean # weighted mean = 6.95
# or #
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean
# use np.average() to get the weighted mean of NumPy arrays or Pandas Series #
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean
# or #
wmean = np.average(z, weights=w)
wmean
# or #
(w * y).sum() / w.sum()
# calculate weighted mean with missing - return 'nan' # 
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()
np.average(y_with_nan, weights=w)
np.average(z_with_nan, weights=w)

# Harmonic Mean #
# pure Python #
hmean = len(x) / sum(1 / item for item in x)
hmean # harmonic mean = 2.76
# calculate this measure with statistics.harmonic_mean() #
hmean = statistics.harmonic_mean(x)
hmean
# If you have a nan value in a dataset, then it’ll return nan. 
# If there’s at least one 0, then it’ll return 0. 
# If you provide at least one negative number, then you’ll get statistics.StatisticsError
statistics.harmonic_mean(x_with_nan) # returns 'nan'
statistics.harmonic_mean([1, 0, 2]) # returns '0'
statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError
# calculate this measure use scipy.stats.hmean() #
scipy.stats.hmean(y)
scipy.stats.hmean(z)
# if your dataset contains nan, 0, a negative number, or anything but positive numbers, then you’ll get a ValueError
scipy.stats.hmean([1, math.nan ,2]) # error
scipy.stats.hmean([1, 0 ,2]) # returns '0.0'
scipy.stats.hmean([1, -7 ,2]) # error

# Geometric Mean #
# pure Python #
gmean = 1
for item in x:
    gmean *= item
gmean **= 1 / len(x)
gmean # gmean = 4.68
# use statistics.geometric_mean() #
gmean = statistics.geometric_mean(x)
gmean
# run with missing - return 'nan' # 
gmean = statistics.geometric_mean(x_with_nan)
gmean
# run with scipy.stats.gmean() #
scipy.stats.gmean(y)
scipy.stats.gmean(z)

# Median #
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])
median_ # median = 4
# run with statistics.median() #
median_ = statistics.median(x)
median_
#  run without the last item 28.0 #
median_ = statistics.median(x[:-1])
median_ # median = 3.25
# If the number of elements is odd, then there’s a single middle value, so these functions behave just like median().
statistics.median_low(x[:-1]) # shown as 2.5
# If the number of elements is even, then there are two middle values. In this case, median_low() returns the lower and median_high() the higher middle value.
statistics.median_high(x[:-1]) # shwon as 4
# run with missing - valid#
statistics.median(x_with_nan) # shown as 6.0
statistics.median_low(x_with_nan) # shown as 4
statistics.median_high(x_with_nan) # shown as 8.0
# run with np.median() #
median_ = np.median(y)
median_ # shown as 4.0
median_ = np.median(y[:-1])
median_ # shown as 3.25
# run with missing and ignore missing #
np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])
# run with pandas with missing - ignore 'nan' by default #
z.median()
z_with_nan.median()

# Mode #
# pure Python #
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_ # mode = 2
# use statistics.mode() and statistics.multimode() #
# mode() returned a single value, while multimode() returned the list that contains the result #
#  If there’s more than one modal value, then mode() raises StatisticsError, while multimode() returns the list with all modes #
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # actually shown as 12 
statistics.multimode(v) # shown as 12, 15
# run with missing #
statistics.mode([2, math.nan, 2]) # shwon as 2
statistics.multimode([2, math.nan, 2]) # shown as 2 
statistics.mode([2, math.nan, 0, math.nan, 5]) # shown as 'nan'
statistics.multimode([2, math.nan, 0, math.nan, 5]) # shown as 'nan'
# use scipy.stats.mode() #
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u) # ModeResult(mode=array([2]), count=array([2]))
mode_
mode_ = scipy.stats.mode(v) # ModeResult(mode=array([12]), count=array([3]))
mode_
# get the mode and its number of occurrences as NumPy arrays with dot notation # 
mode_.mode
mode_.count
# use pandas #
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode() # shown as 0, 2
v.mode()
w.mode()


### Measures of Variability ###
# Variance #
# pure Python #
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_ # variance = 123.20
# use statistics.variance() #
statistics.variance(x_with_nan) # return 'nan'
# use np.var() or the corresponding method .var()#
var_ = np.var(y, ddof=1)
var_
var_ = y.var(ddof=1)
var_
# run with missing - return 'nan' #
np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)
# run with misisng and ignore missing#
np.nanvar(y_with_nan, ddof=1)
# use pandas - ignore 'nan' by default #
z.var(ddof=1)
z_with_nan.var(ddof=1)

# Standard Deviation #
# pure Python#
std_ = var_ ** 0.5
std_ # sd = 11.10
# use statistics.stdev() #
std_ = statistics.stdev(x)
std_
# use Numpy #
np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
# run with missing and ignore missing # 
np.nanstd(y_with_nan, ddof=1)
# use pandas #
z.std(ddof=1)
z_with_nan.std(ddof=1)

# Skewness #
# pure python #
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_ # shown as 1.95 > 0, x has right-side tail
# use scipy.stats.skew() #
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False) # return 'nan' 
# use .skew() #
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan) # return 'nan' 
z.skew() # shown as 1.95
z_with_nan.skew()

# Percentiles #
# divide your data into several intervals use statistics.quantiles() #
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2) # shown as 8.0
statistics.quantiles(x, n=4, method='inclusive') # shown as 0.1, 8.0, 21.0 
# find the 5th and 95th percentiles #
y = np.array(x)
np.percentile(y, 5) # is -3.44
np.percentile(y, 95) # is 34.92
np.percentile(y, [25, 50, 75]) # array ([ 0.1,  8. , 21. ])
np.median(y) # median is 8.0
# run with missing use np.nanpercentile() # 
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan # array([-5. , -1.1,  nan,  0.1,  2. ,  8. , 12.8, 21. , 25.8, 41. ])
np.nanpercentile(y_with_nan, [25, 50, 75]) # array([ 0.1,  8. , 21. ])
# use Numpy quantile() and nanquantile() #
np.quantile(y, 0.05)
np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])
# use pandas # 
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

# Ranges #
# use  np.ptp() # 
np.ptp(y) # 46.0
np.ptp(z)
np.ptp(y_with_nan) # return 'nan' 
np.ptp(z_with_nan) # return 'nan' 
# show min or max # 
np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()
# show interquartile range - the difference between the first and third quartile #
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0] # 20.9
quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

### Summary of Descriptive Statistics ### 
# quickly get descriptive statistics with a single function or method call # 
result = scipy.stats.describe(y, ddof=1, bias=False)
result
result.nobs # number of observations or elements in your dataset n = 9
result.minmax[0]  # Min
result.minmax[1]  # Max
result.mean
result.variance
result.skewness
result.kurtosis
# use pandas # 
result = z.describe()
result
result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

### Measures of Correlation Between Pairs of Data ###
# use NumPy arrays and Pandas Series # 
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

# Covariance # 
# pure python #
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy # coef is 19.95
# use Numpy #
cov_matrix = np.cov(x_, y_)
cov_matrix
x_.var(ddof=1) # 38.5
y_.var(ddof=1) # 13.91
cov_xy = cov_matrix[0, 1]
cov_xy
cov_xy = cov_matrix[1, 0]
cov_xy
# use pandas # 
cov_xy = x__.cov(y__)
cov_xy # 19.95
cov_xy = y__.cov(x__)
cov_xy # 19.95

# Correlation Coefficient #
# pure python #
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r # 0.86
# use scipy.stats calculates the correlation coefficient and the 𝑝-value # 
r, p = scipy.stats.pearsonr(x_, y_)
r # 0.86
p # 5.12
# use np.corrcoef() #
corr_matrix = np.corrcoef(x_, y_)
corr_matrix
r = corr_matrix[0, 1]
r
r = corr_matrix[1, 0]
r
# use  scipy.stats.linregress() #
scipy.stats.linregress(x_, y_)
# to access particular values from the result of linregress() # 
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r
# use pandas #
r = x__.corr(y__)
r
r = y__.corr(x__)
r

### Working With 2D Data ###
# Axes # 
# creating a 2D NumPy array #
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a
np.mean(a) # 5.4
a.mean()
np.median(a) # 2.0
a.var(ddof=1) # 53.40
# mean for rows #
np.mean(a, axis=0) # array([6.2, 8.2, 1.8])
a.mean(axis=0)
# mean for columns # 
np.mean(a, axis=1) # array([ 1.,  2.,  5., 13.,  6.])
a.mean(axis=1)
# median and variance #
np.median(a, axis=0) # for cols: array([4., 3., 1.])
np.median(a, axis=1) # for rows: array([1., 2., 4., 8., 1.])
a.var(axis=0, ddof=1) # for cols: array([ 37.2, 121.2,   1.7])
a.var(axis=1, ddof=1) # for rows: array([  0.,   1.,  13., 151.,  75.])
# gmean use scipy #
scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0) # array([4., 3.73719282, 1.51571657])
scipy.stats.gmean(a, axis=1)
scipy.stats.gmean(a, axis=None) # for entire dataset 
# statistics summary # 
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

# DataFrames #
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df
df.mean() # return col mean
df.mean(axis=1) # return row mean
df.var() # return col variance
df.var(axis=1) # return row variance 
# for column 'A' #
df['A']
df['A'].mean() # mean = 6.2
df['A'].var() # variance = 37.20
# use a DataFrame as a NumPy array, get all data from a DataFrame with .values or .to_numpy() #
df.values
df.to_numpy() # without row and column labels
# statistics summary for all columns #
df.describe()
# access each item of the summary
df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

### Visualizing Data ###
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Box plots # 
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

# Histograms #
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

# Pie Charts # 
x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

# Bar Charts #
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# X-Y Plots # 
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

# Heatmaps #
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()
