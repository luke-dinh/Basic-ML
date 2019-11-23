from sklearn import linear_model
import numpy as np
x = np.array([[6,4,8,5]]).T
y = np.array([7,5,9,4])
one = np.ones((x.shape[0],1))
Xbar = np.concatenate((one,x),axis =1)
t = linear_model.LinearRegression(fit_intercept = False)
t.fit(Xbar,y)
print('intercept:', t.intercept_)
print('slope',t.coef_)




