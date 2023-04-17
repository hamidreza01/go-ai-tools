package main

import (
	"fmt"
	"math"
)

type SVM struct {
	weights []float64
	bias    float64
	c       float64
}

func NewSVM(c float64) *SVM {
	return &SVM{c: c}
}

func (svm *SVM) fit(X [][]float64, y []float64, epochs int, alpha float64) {
	nSamples, nFeatures := len(X), len(X[0])
	svm.weights = make([]float64, nFeatures)
	svm.bias = 0.0

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < nSamples; i++ {
			prediction := svm.predict(X[i])
			if y[i]*prediction < 1 {
				for j := 0; j < nFeatures; j++ {
					svm.weights[j] += alpha * (y[i]*X[i][j] - svm.c*svm.weights[j])
				}
				svm.bias += alpha * y[i]
			} else {
				for j := 0; j < nFeatures; j++ {
					svm.weights[j] += alpha * (-svm.c * svm.weights[j])
				}
			}
		}
	}
}

func (svm *SVM) predict(X []float64) float64 {
	z := 0.0
	for i := 0; i < len(X); i++ {
		z += svm.weights[i] * X[i]
	}
	z += svm.bias
	if z >= 0 {
		return 1.0
	}
	return -1.0
}
