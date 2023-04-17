package main

import (
    "fmt"
    "math"
    "math/rand"
)

type NeuralNetwork struct {
    numInputs int
    numOutputs int
    numHiddenLayers int
    numNeuronsPerLayer int
    weights [][]float64
}

func NewNeuralNetwork(numInputs int, numOutputs int, numHiddenLayers int, numNeuronsPerLayer int) *NeuralNetwork {
    nn := &NeuralNetwork{
        numInputs: numInputs,
        numOutputs: numOutputs,
        numHiddenLayers: numHiddenLayers,
        numNeuronsPerLayer: numNeuronsPerLayer,
        weights: make([][]float64, numHiddenLayers+1),
    }

    for i := 0; i < len(nn.weights); i++ {
        if i == 0 {
            nn.weights[i] = make([]float64, nn.numInputs*numNeuronsPerLayer)
        } else if i == numHiddenLayers {
            nn.weights[i] = make([]float64, nn.numOutputs*numNeuronsPerLayer)
        } else {
            nn.weights[i] = make([]float64, nn.numNeuronsPerLayer*numNeuronsPerLayer)
        }

        for j := 0; j < len(nn.weights[i]); j++ {
            nn.weights[i][j] = rand.Float64() * 2 - 1 // Initialize weights to random values between -1 and 1
        }
    }

    return nn
}

func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func (nn *NeuralNetwork) forward(inputs []float64) []float64 {
    activations := make([]float64, nn.numNeuronsPerLayer)

    for i := 0; i < nn.numNeuronsPerLayer; i++ {
        sum := 0.0
        for j := 0; j < nn.numInputs; j++ {
            sum += inputs[j] * nn.weights[0][i*nn.numInputs+j]
        }
        activations[i] = sigmoid(sum)
    }

    for i := 1; i < nn.numHiddenLayers+1; i++ {
        nextActivations := make([]float64, nn.numNeuronsPerLayer)

        for j := 0; j < nn.numNeuronsPerLayer; j++ {
            sum := 0.0
            for k := 0; k < nn.numNeuronsPerLayer; k++ {
                sum += activations[k] * nn.weights[i][j*nn.numNeuronsPerLayer+k]
            }
            nextActivations[j] = sigmoid(sum)
        }

        activations = nextActivations
    }

    outputs := make([]float64, nn.numOutputs)

    for i := 0; i < nn.numOutputs; i++ {
        sum := 0.0
        for j := 0; j < nn.numNeuronsPerLayer; j++ {
            sum += activations[j] * nn.weights[nn.numHiddenLayers][i*nn.numNeuronsPerLayer+j]
        }
        outputs[i] = sigmoid(sum)
    }

    return outputs
}
