import numpy as np

class LinearRegressor():
    def __init__(self, x, y) -> None:
        self.x, self.y = x, y
    def solve(self):
        self.w = np.linalg.lstsq(self.x, self.y, rcond=None)[0]
    def predict(self, new_x):
        return new_x.dot(self.w)

if __name__ == '__main__':
    A = np.random.random((59,))
    B = np.random.random((70, 59))
    C = np.dot(B, A) + np.random.random(np.dot(B, A).shape)
    W = np.linalg.lstsq(B, C)[0]
    print(A)
    print(W)