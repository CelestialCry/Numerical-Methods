# Løsning av SVD dekomposisjon
import numpy as np

def gs(ls):
    B = [ls[0]/np.linalg.norm(ls[0])]
    for v in ls[1:]:
        u = v
        for w in B:
            u = u - np.dot(v,w)/np.dot(w,w)*w
        B.append(u/np.linalg.norm(u))
    return B

def svd(A):
    ATA = np.transpose(A)@A
    singSq, eigVec = np.linalg.eigh(ATA)

    V = np.transpose(np.array(eigVec))

    ls = list(zip(singSq, V))
    sortert = sorted(ls, key = lambda a: a[0], reverse = True)
    V, singSq = [], []

    for (σ, ν) in sortert:
        singSq.append(σ)
        V.append(ν)

    print(np.transpose(V))

    def rank(ls): # We need this function to find the rank
        dimension = len(ls)
        nullity = 0
        for el in ls:
            if el == 0:
                nullity += 1
        return dimension - nullity

    matrixRank = rank(singSq)
    Σ = np.full(A.shape, 0)
    for i in range(matrixRank):
        Σ[i][i] = singSq[i]**(1/2)


    Ur = [(lambda a: a/np.linalg.norm(a))(A@v) for v in V]
    # Ur = []
    # for i in range(len(V)):
    #     print(i, V[i])
    #     nv = A@V[i]
    #     Ur.append(nv/np.linalg.norm(nv))

    

    while len(Ur) < A.shape[0]:
        Ur.append(np.full(U[0].shape, 1))
    U = np.array(gs(Ur))
    return Σ, np.transpose(V), U
    # I need to orthogonalize the rest of Ur matrix

M = np.array([[1,2,0],[2,1,0],[0,0,1]])
S, Vᵀ, U = svd(M)
print(f"S:\n{S}")
print(f"Vᵀ:\n{Vᵀ}")
print(f"U:\n{U}")

print(U@S@Vᵀ)

# U = np.array([[np.sqrt(1/2), np.sqrt(1/2), 0], [0, 0, 1], [-np.sqrt(1/2), np.sqrt(1/2), 0]])
# Σ = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 1]])
# Vᵀ = np.array([[np.sqrt(1/2), np.sqrt(1/2), 0], [0, 0, 1], [np.sqrt(1/2), -np.sqrt(1/2), 0]])

# print(np.transpose(Vᵀ)@Σ@U)