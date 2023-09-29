import numpy as np
from collections import Counter
from functools import partial
from math import gcd
from numba import jit
from itertools import product
from numba.typed import List

@jit(nopython=True)
def e_m(x,m):
    return np.exp((x%m)*2*np.pi*1j/m)

# Generate all possible n x n matrices with entries modulo m
def NxNMatsOverM(n,m):
    return [np.array(i).reshape(n,n) for i in list(product(list(range(m)),repeat=n**2))]
# Find units (invertible matrices) among the given matrices
def Units(mats,m):
    return [mat for mat in mats if egcd(round(np.linalg.det(mat))%m,m)[0]==1]
# Generate all matrices of sizes from 1x1 to n x n modulo m
def AllMatsGenerator(n,m):
    mats=[]
    for i in range(n):
        mats.append(NxNMatsOverM(i+1,m))
    return mats
# Generate invertible matrices for each matrix size
def UnitsGenerator(allMats,m):
    invertibles=[]
    for mats in allMats:
        invertibles.append(Units(mats,m))
    return invertibles
# Find the index of the first row/column that contains only zeros except on the diagonal
def OrthoRowCol(mat):
    for i in range(mat.shape[0]):
        flag = True
        for k in range(mat.shape[0]):
            if (mat[i,k]!=0 or mat[k,i]!=0) and (i!=k):
                flag=False
        if flag:
            return i
    return -1
#calculates jacobi symbol
def jacobi(a, n):
    a=a%n
    if a==0:
        return 0
    assert(n > a > 0 and n%2 == 1)
    t = 1
    while a != 0:
        while a % 2 == 0:
            a /= 2
            r = n % 8
            if r == 3 or r == 5:
                t = -t
        a, n = n, a
        if a % 4 == n % 4 == 3:
            t = -t
        a %= n
    if n == 1:
        return t
    else:
        return 0
# Euclidean algorithm to compute the greatest common divisor (GCD)
@jit(nopython=True)
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

# Calculate the modular inverse of a modulo m
@jit(nopython=True)
def modinv(a, m):
    a=a%m
    g, x, y = egcd(a, m)
    if g != 1:
        print(a,m)
        raise Exception('modular inverse does not exist')
    else:
        return x % m
    
# Calculate the inverse of a matrix modulo m
@jit(nopython=True,parallel=True)
def inverse(mat,m):
    mat=mat.astype("float")
    det = (np.linalg.det(mat))
    adj= (np.linalg.inv(mat))*det
    fact=modinv(round(det%m),m)
    return np.rint(adj*fact)%m
#Calculates Matrix Gauss Sum given data about values
def GaussSumData(data,a):
    for i in range(len(data[0])):
        if ArrayInList(a,data[0][i]):
            return data[1][i]
#Calculats Quadratic Gauss Sum
def QuadraticGaussSum(a,m):
    total=0
    for i in range(m):
        total+= e_m(a*i**2,m)
    return total
# Function for computing a matrix Gauss sum given a conjugacy class
def CClassGaussSum(cclass,m,prevData,allMats):
    dim=cclass[0].shape[0]
    for a in cclass:
        ortho =OrthoRowCol(a)
        if ortho !=-1:
            a_kk= a[ortho,ortho]
            reducedMat = np.delete(a,(ortho),axis=0)
            reducedMat = np.delete(reducedMat,ortho,axis=1)
            orthoComponent = QuadraticGaussSum(np.array([a_kk]),m)
            if reducedMat.size==1:
                reducedSum = QuadraticGaussSum(reducedMat.item(0),m)
            else:
                reducedSum = GaussSumData(prevData,reducedMat)
            crossterms=0
            sols = product(list(range(m)),repeat=(dim-1)*2)
            for sol in sols:
                lin=0
                for i in range(dim-1):
                    for j in range(dim-1):
                        term = reducedMat[i,j]
                        if i ==j:
                            term+=a_kk
                        lin+= term*sol[j]*sol[i+dim-1]
                crossterms+=e_m(lin,m)
            value = orthoComponent*reducedSum*crossterms
            return value
    return GaussSumBruteForce(allMats,cclass[0],m)
#Efficient (using reduction formula) Matric Gauss Sum
def GaussSumEfficient(allMats,A,invertibles,m,prevData):
    dim=A.shape[0]
    count=0
    for invert in invertibles:
        if count >=10000:
            return GaussSumBruteForce(allMats,A,m)
        count+=1
        a=(invert@A@inverse(invert,m))
        ortho =OrthoRowCol(a)
        if ortho !=-1:
            a_kk= a[ortho,ortho]
            reducedMat = np.delete(a,(ortho),axis=0)
            reducedMat = np.delete(reducedMat,ortho,axis=1)
            orthoComponent = QuadraticGaussSum(np.array([a_kk]),m)
            if reducedMat.size==1:
                reducedSum = QuadraticGaussSum(reducedMat.item(0),m)
            else:
                reducedSum = GaussSumData(prevData,reducedMat)
            crossterms=0
            sols = product(list(range(m)),repeat=(dim-1)*2)
            for sol in sols:
                lin=0
                for i in range(dim-1):
                    for j in range(dim-1):
                        term = reducedMat[i,j]
                        if i ==j:
                            term+=a_kk
                        lin+= term*sol[j]*sol[i+dim-1]
                crossterms+=e_m(lin,m)
            value = orthoComponent*reducedSum*crossterms
            return value
    return GaussSumBruteForce(allMats,A,m)

#Bruteforce Matrix Gauss Sum
def GaussSumBruteForce(mats,a,m):
    total=0
    a=np.transpose(a)
    for mat in mats:
        if a.size==1:
            total+=e_m((a@mat@mat)[0],m)
        else:
            total+= e_m(np.trace(a@mat@mat),m)
    return total
#Get index of arr in lst
@jit(nopython=True,parallel=True)
def IndexArrInList(arr,lst):
    i=0
    for a in lst:
        if np.array_equal(a,arr):
            return i
        i+=1
    return -1
#Checks whether arr is in lst
@jit(nopython=True,parallel=True)
def ArrayInList(arr,lst):
    for a in lst:
        if np.array_equal(a,arr):
            return True
    return False
#counts instances of arr in lst
def ArrayCount(arr,lst):
    total=0
    for a in lst:
        if np.array_equal(a,arr):
            total+=1
    return total
#Calculates Conjugacy Classes
@jit(nopython=True,parallel=True)
def ConjugacyClass(a,invertibles,m):
    cclass=[a]
    length=0
    for unit in range(len(invertibles)):
        if unit%15000==0 and len(invertibles)%len(cclass)==0:
            if len(cclass)==length:
                return cclass
            else:
                length=len(cclass)
        mat= np.rint((invertibles[unit].astype('float')@a.astype('float')@inverse(invertibles[unit].astype('float'),m)).astype('float')) %m
        if not ArrayInList(mat,cclass):
            cclass.append(mat.astype("int"))
    return cclass
#Removes list2 from list1
@jit(nopython=True,parallel=True)
def RemoveListFromList(list1,list2):
    i=0
    list2=list2[:]
    while len(list2)>0:
        index=IndexArrInList(list1[i],list2)
        if index!=-1:
            list1.pop(i)
            list2.pop(index)
        else:
            i+=1
    return list1
#Calculates conjectured value (not valid for all a)
def GaussSumExplicitForm2x2(a,m,squares):
    trace=round(np.trace(a))%m
    det = round(np.linalg.det(a))%m
    if trace==0:
        return m**3
    return m**2*egcd(trace,m)[0]*jacobi(-1*det, m)
#Calculates Conjugacy Classes and solutions for given matrix size and modulus
def ConjugacyClassSols(allMats,prevData,invertibles,m):
    invertCopy = allMats[:]
    cclasses=[]
    sols=[]
    sols2=[]
    while len(invertCopy)>0:
        mat=invertCopy[0]
        print("conj calc")
        cclass=ConjugacyClass(mat,invertibles,m)
        invertCopy=RemoveListFromList(invertCopy,cclass)
        cclasses.append(cclass)
        print("gauss sum calc")
        sol =CClassGaussSum(cclass,m,prevData,allMats)
        sols.append(CClassGaussSum(cclass,m,prevData,allMats))
        print(mat,"sol",sol,"det",np.linalg.det(mat)%m,"trace",np.trace(mat)%m)
        print(len(invertCopy))
    return [cclasses,sols]
def CalcAllGaussSumsUpToN(n,m):
    data=[[]]
    for i in range(2,n+1):
        mats= NxNMatsOverM(i,m)
        units= Units(mats,m)
        prevData=data[i-2]
        sols=ConjugacyClassSols(mats,prevData,units,m)
        data.append(sols)
    conj=True
    for i in range(len(data[1][0])):
        if np.abs(GaussSumExplicitForm2x2(data[1][0][i][0],m,m) -data[1][1][i])>0.01:
            print("False",data[1][0][i][0])
            conj=False
    if conj:
        print("verified")
    return data
