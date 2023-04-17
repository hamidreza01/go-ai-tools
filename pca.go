package main

import (
    "fmt"
    "math"
)

type PCA struct {
    nComponents int
}

func NewPCA(nComponents int) * PCA {
    return &PCA {
        nComponents: nComponents
    }
}

func(pca * PCA) fit(X[][] float64)([][] float64, [] float64) {
    nSamples,
    nFeatures: = len(X),
    len(X[0])


    mean: = make([] float64, nFeatures)
    for i: = 0;i < nSamples;i++{
        for j: = 0;j < nFeatures;j++{
            mean[j] += X[i][j] / float64(nSamples)
        }
    }

    for i: = 0;i < nSamples;i++{
        for j: = 0;j < nFeatures;j++{
            X[i][j] -= mean[j]
        }
    }


    cov: = make([][] float64, nFeatures)
    for i: = range cov {
        cov[i] = make([] float64, nFeatures)
    }
    for i: = 0;i < nFeatures;i++{
        for j: = 0;j < nFeatures;j++{
            for k: = 0;k < nSamples;k++{
                cov[i][j] += X[k][i] * X[k][j] / float64(nSamples - 1)
            }
        }
    }


    eigenvalues,
    eigenvectors: = eig(cov)

        idx: = make([] int, nFeatures)
    for i: = range idx {
        idx[i] = i
    }
    for i: = 0;i < nFeatures;i++{
        for j: = i + 1;j < nFeatures;j++{
            if eigenvalues[j] > eigenvalues[i] {
                eigenvalues[i], eigenvalues[j] = eigenvalues[j], eigenvalues[i]
                idx[i], idx[j] = idx[j], idx[i]
            }
        }
    }


    eigenvectors = eigenvectors[: pca.nComponents]


    transformed: = make([][] float64, nSamples)
    for i: = 0;i < nSamples;i++{
        transformed[i] = make([] float64, pca.nComponents)
        for j: = 0;j < pca.nComponents;j++{
            for k: = 0;k < nFeatures;k++{
                transformed[i][j] += X[i][k] * eigenvectors[j][k]
            }
        }
    }

    return transformed,
    eigenvalues
}

func eig(A[][] float64)([] float64, [][] float64) {
    n: = len(A)
    Q: = make([][] float64, n)
    for i: = range Q {
        Q[i] = make([] float64, n)
    }
    V: = make([][] float64, n)
    for i: = range V {
        V[i] = make([] float64, n)
    }
    for i: = 0;i < n;i++{
        for j: = 0;j < n;j++{
            Q[i][j] = A[i][j]
            if i == j {
                V[i][j] = 1.0
            }
        }
    }

    for i: = 0;i < 100;i++{
        p,
        q: = 0,
        0
        max: = math.Abs(Q[0][1])
        for j: = 0;j < n;j++{
            for k: = j + 1;k < n;k++{
                if math.Abs(Q[j][k]) > max {
                    max = math.Abs(Q[j][k])
                    p, q = j, k
                }
            }
        }

        if max < 1e-10 {
            break
        }


        theta: = 0.5 * math.Atan2(2 * Q[p][q], Q[q][q] - Q[p][p])


            c,
        s: = math.Cos(theta),
        math.Sin(theta)
        R: = make([][] float64, n)
        for i: = range R {
            R[i] = make([] float64, n)
            R[i][i] = 1.0
        }
        R[p][p],
        R[q][q] = c,
        c
        R[p][q],
        R[q][p] = -s,
        s
        Qp: = make([][] float64, n)
        for i: = range Qp {
            Qp[i] = make([] float64, n)
        }
        Vp: = make([][] float64, n)
        for i: = range Vp {
            Vp[i] = make([] float64, n)
        }
        for i: = 0;i < n;i++{
            for j: = 0;j < n;j++{
                for k: = 0;k < n;k++{
                    Qp[i][j] += Q[i][k] * R[k][j]
                    Vp[i][j] += V[i][k] * R[k][j]
                }
            }
        }
        Q,
        V = Qp,
        Vp
    }

    eigenvalues: = make([] float64, n)
    eigenvectors: = make([][] float64, n)
    for i: = 0;i < n;i++{
        eigenvalues[i] = Q[i][i]
        eigenvectors[i] = make([] float64, n)
        for j: = 0;j < n;j++{
            eigenvectors[i][j] = V[j][i]
        }
    }

    return eigenvalues,
    eigenvectors
}
