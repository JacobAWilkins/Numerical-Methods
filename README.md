# Numerical Methods
### Run
```
python NumericalMethods.py
```

### Solve Linear Systems of Equations
Evaluates/Finds the least squared solutions of Ax=b for x with 3 different methods.
A is an __m x n__ matrix, x and b are __1 x n__ column vectors.
Uses A and b from the svd-data.csv dataset.

1. Singular Value Decomposition (A=U.S.V^t) regardless of A's shape or rank
```
x=sum(i=0, r){ (u_i^t.b / s_i).v_i } where r is the effective rank of A
```
2. Normal Equations without inverse of A^t.A
```
A^t.A.x = A^t.b
```
3. Normal Equations with inverse of A^t.A
```
x = (A^t.A)^-1.A^t.b
```

Finds the 2-norm of the residual vector for each.

Each method results in a different value for x. The value of x used to produce b is
```
x = [1, 1, 1, 1, 1, 1, 1]^t
```
### Conclusion
The condition number of AtA is very high which mean any operations that used AtA
are ill-conditioned, thus, inaccurate. This is why the solutions of x vary so much.
