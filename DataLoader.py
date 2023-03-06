from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split


X, y = datasets.fetch_openml( "mnist_784", version=1, return_X_y=True, as_frame=False)
print(X,y)