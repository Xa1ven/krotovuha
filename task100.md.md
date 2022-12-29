1. Import the numpy package under the name `np` (★☆☆)

import numpy as np

2. Print the numpy version and the configuration (★☆☆)

print(np.__version__)
np.show_config()

3. Create a null vector of size 10 (★☆☆)

Z = np.zeros(10)
print(Z)

6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

Z = np.zeros(10)
Z[4] = 1
print(Z)
```
7. Create a vector with values ranging from 10 to 49 (★☆☆)

Z = np.arange(10,50)
print(Z)

8. Reverse a vector (first element becomes last) (★☆☆)

Z = np.arange(50)
Z = Z[::-1]
print(Z)

9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

Z = np.arange(9).reshape(3, 3)
print(Z)

10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)

nz = np.nonzero([1,2,0,0,4,0])
print(nz)

11. Create a 3x3 identity matrix (★☆☆)

Z = np.eye(3)
print(Z)

12. Create a 3x3x3 array with random values (★☆☆)

Z = np.random.random((3,3,3))
print(Z)

13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)

14. Create a random vector of size 30 and find the mean value (★☆☆)

Z = np.random.random(30)
m = Z.mean()
print(m)

15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)

17. What is the result of the following expression? (★☆☆)
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1

print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

print(np.unravel_index(99,(6,7,8)))

25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)

28. What are the result of the following expressions? (★☆☆)
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

30. How to find common values between two arrays? (★☆☆)

Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))

40. Create a random vector of size 10 and sort it (★★☆)

Z = np.random.random(10)
Z.sort()
print(Z)

42. Consider two random array A and B, check if they are equal (★★☆)

A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)
equal = np.array_equal(A,B)
print(equal)

44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

53. How to convert a float (32 bits) array into an integer (32 bits) in place?

Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)

58. Subtract the mean of each row of a matrix (★★☆)

X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
Y = X - X.mean(axis=1).reshape(-1, 1)
print(Y)
