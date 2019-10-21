#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:38:50 2019

@author: Jacob Wilkins
@copyright: GNU General Public Licence

Finds the least squares solution (x) to a system of equations using
Singular Value Decomposition (A=U.sigma.V^t) regardless of A's shape or rank,
the normal equations A^t.A.x = A^t.b, x = (A^t.A)^-1.A^t.b,
and x=sum_i=0_to_r((u_i^t.b/sigma_i).v_i) where r is the effective rank of A.
Finds the 2-norm of the residual vector for each.

Each method results in a different value for x. The value of x used to produce b
is x = [1, 1, 1, 1, 1, 1, 1]^t

The condition number of AtA is very high which mean any operations that used AtA
are ill-conditioned, thus, inaccurate. This is why the solutions of x vary so much.
"""

from numpy import linalg as LA, dot, size, zeros, genfromtxt

# get A and b from dataset
def getAb(data): return data[:, 0:size(data, 1)-1], data[:, size(data, 1)-1]

# A and b from dataset
A, b = getAb(genfromtxt('svd-data.csv', delimiter = ','));

# singular value decomposition of A
u, s, v = LA.svd(A)

# x = AtA^-1 * Atb
def getX1(): return dot(dot(LA.inv(dot(A.T, A)), A.T), b)

# solve for x: AtA * x = Atb; np.linalg.solve(At*A, At*b)
def getX2(): return LA.solve(dot(A.T, A), dot(A.T, b))

# use SVD and rank to solve for x
def getX3(rank, x = zeros((size(A, 1)))):
    for r in range(0, rank): x += dot((dot(u[:, r].T, b) / s[r]), v[:, r])
    return x

# condition number of AtA: ||cond(AtA^-1)||2 * ||cond(AtA)||2
def getCondAtA(AtA): return LA.norm(LA.inv(AtA), 2) * LA.norm(AtA, 2)

# get 2-norm of residual vector
def getRN(x): return LA.norm(b - dot(A, x), 2)

# get rank of A; count number of singular values greater than 0.1
def getRank(rank = 0):
    for sv in s: 
        if sv >= 0.1: rank += 1
    return rank

def main():
    # print x1 & 2-norm of x1
    print("i. x = AtA^-1 * Atb =", getX1(), "\n") 
    print("ii.2-norm of residual vector = ||b - Ax||2 = ||r||2 = %f\n" % getRN(getX1()))
    
    # print x2 & 2-norm of x2
    print("iii. x = np.linalg.solve(At*A, At*b) =", getX2(), "\n")
    print("iv.2-norm of residual vector = ||b - Ax||2 = ||r||2 = %f\n" % getRN(getX2()))
    
    # print singular values
    print("v. Singular values of A = ", s, "\n")
    
    # print effective rank of A
    print("vi. Effective rank of A =  %d\n" % getRank())
    
    # print x3 & 2-norm of b - A * x3
    print("vii. x =", getX3(getRank()), "\n")
    
    print("viii. ||b - Ax||2 = ||r||2 = %f\n" % getRN(getX3(getRank())))
    
    # print condition number of AtA
    print("ix. Condition number of AtA:", getCondAtA(dot(A.T, A)))

if __name__ == "__main__":
    main()
