# Numerical Methods
### Solve Linear Systems of Equations
Evaluates/Finds the least squared solutions of Ax=b for x with 3 different methods. Uses A and b from the svd-data.csv dataset.

1. Singular Value Decomposition (A=U.sigma.V^t) regardless of A's shape or rank
```
x=sum_i=0_to_r((u_i^t.b/sigma_i).v_i) where r is the effective rank of A
```
2. Normal Equations without inverse of A^t.A
```
A^t.A.x = A^t.b
```
3. Normal Equations with inverse of A^t.A
```
x = (A^t.A)^-1.A^t.b
```

### Run
```
python NumericalMethods.py
```
