package main

import (
    "math"
)

type RNN struct {
    inputSize int
    hiddenSize int
    outputSize int

    Wxh[][] float64
    Whh[][] float64
    Why[][] float64

    bh[] float64
    by[] float64

    hPrev[] float64
}

func NewRNN(inputSize int, hiddenSize int, outputSize int) * RNN {
    rnn: = new(RNN)
    rnn.inputSize = inputSize
    rnn.hiddenSize = hiddenSize
    rnn.outputSize = outputSize

    rnn.Wxh = make([][] float64, hiddenSize)
    rnn.Whh = make([][] float64, hiddenSize)
    rnn.Why = make([][] float64, outputSize)

    for i: = 0;i < hiddenSize;i++{
        rnn.Wxh[i] = make([] float64, inputSize)
        rnn.Whh[i] = make([] float64, hiddenSize)
        rnn.bh = append(rnn.bh, 0)
    }

    for i: = 0;i < outputSize;i++{
        rnn.Why[i] = make([] float64, hiddenSize)
        rnn.by = append(rnn.by, 0)
    }

    rnn.hPrev = make([] float64, hiddenSize)

    rnn.Randomize()

    return rnn
}

func(rnn * RNN) Randomize() {
    for i: = 0;
    i < rnn.hiddenSize;
    i++{
        for j: = 0;j < rnn.inputSize;j++{
            rnn.Wxh[i][j] = Rand(-0.5, 0.5)
        }

        for j: = 0;j < rnn.hiddenSize;j++{
            rnn.Whh[i][j] = Rand(-0.5, 0.5)
        }

        for j: = 0;j < rnn.outputSize;j++{
            rnn.Why[j][i] = Rand(-0.5, 0.5)
        }

        rnn.bh[i] = 0
    }

    for i: = 0;
    i < rnn.outputSize;
    i++{
        rnn.by[i] = 0
    }

    rnn.hPrev = make([] float64, rnn.hiddenSize)
}

func(rnn * RNN) Train(inputs[][] float64, targets[][] float64, lr float64) {
    N: = len(inputs)
    xs: = make([][] float64, N + 1)
    hs: = make([][] float64, N + 1)
    ys: = make([][] float64, N + 1)
    ps: = make([][] float64, N + 1)

        hs[0] = make([] float64, rnn.hiddenSize)
    ys[0] = make([] float64, rnn.outputSize)

    for t: = 0;t < N;t++{
        xs[t + 1] = make([] float64, rnn.inputSize)
        copy(xs[t + 1], inputs[t])

        for i: = 0;i < rnn.hiddenSize;i++{
            hs[t + 1] = append(hs[t + 1], 0)
            for j: = 0;j < rnn.inputSize;j++{
                hs[t + 1][i] += rnn.Wxh[i][j] * xs[t + 1][j]
            }

            for j: = 0;j < rnn.hiddenSize;j++{
                hs[t + 1][i] += rnn.Whh[i][j] * hs[t][j]
            }

            hs[t + 1][i] = math.Tanh(hs[t + 1][i] + rnn.bh[i])
        }
        for i: = 0;i < rnn.outputSize;i++{
            ys[t + 1] = append(ys[t + 1], 0)
            for j: = 0;j < rnn.hiddenSize;j++{
                ys[t + 1][i] += rnn.Why[i][j] * hs[t + 1][j]
            }

            ys[t + 1][i] += rnn.by[i]
        }

        ps[t + 1] = Softmax(ys[t + 1])

        for i: = 0;i < rnn.outputSize;i++{
            for j: = 0;j < rnn.hiddenSize;j++{
                dWhy: = CrossEntropyDerivative(ps[t + 1][i], targets[t][i]) * hs[t + 1][j]
                rnn.Why[i][j] -= lr * dWhy
            }

            dby: = CrossEntropyDerivative(ps[t + 1][i], targets[t][i])
            rnn.by[i] -= lr * dby
        }

        delta: = make([] float64, rnn.hiddenSize)

        for i: = 0;i < rnn.hiddenSize;i++{
            for j: = 0;j < rnn.outputSize;j++{
                delta[i] += CrossEntropyDerivative(ps[t + 1][j], targets[t][j]) * rnn.Why[j][i]
            }

            delta[i] *= (1 - math.Pow(hs[t + 1][i], 2))
        }

        for i: = 0;i < rnn.hiddenSize;i++{
            for j: = 0;j < rnn.inputSize;j++{
                dWxh: = delta[i] * xs[t + 1][j]
                rnn.Wxh[i][j] -= lr * dWxh
            }

            for j: = 0;j < rnn.hiddenSize;j++{
                dWhh: = delta[i] * hs[t][j]
                rnn.Whh[i][j] -= lr * dWhh
            }

            dbh: = delta[i]
            rnn.bh[i] -= lr * dbh
        }

        rnn.hPrev = hs[N]
    }
}
func(rnn * RNN) Forward(inputs[][] float64)[][] float64 {
    N: = len(inputs)
    hs: = make([][] float64, N + 1)
    ys: = make([][] float64, N + 1)
    ps: = make([][] float64, N + 1)
    hs[0] = make([] float64, rnn.hiddenSize)
    ys[0] = make([] float64, rnn.outputSize)

    for t: = 0;t < N;t++{
        x: = make([] float64, rnn.inputSize)
        copy(x, inputs[t])

        for i: = 0;i < rnn.hiddenSize;i++{
            hs[t + 1] = append(hs[t + 1], 0)
            for j: = 0;j < rnn.inputSize;j++{
                hs[t + 1][i] += rnn.Wxh[i][j] * x[j]
            }

            for j: = 0;j < rnn.hiddenSize;j++{
                hs[t + 1][i] += rnn.Whh[i][j] * hs[t][j]
            }

            hs[t + 1][i] = math.Tanh(hs[t + 1][i] + rnn.bh[i])
        }
        for i: = 0;i < rnn.outputSize;i++{
            ys[t + 1] = append(ys[t + 1], 0)
            for j: = 0;j < rnn.hiddenSize;j++{
                ys[t + 1][i] += rnn.Why[i][j] * hs[t + 1][j]
            }

            ys[t + 1][i] += rnn.by[i]
        }

        ps[t + 1] = Softmax(ys[t + 1])
    }

    return ps[1: ]
}
func CrossEntropyDerivative(p, y float64) float64 {
    return p - y
}

func Softmax(xs[] float64)[] float64 {
    ys: = make([] float64, len(xs))
    sum: = 0.0
    for i: = range xs {
        ys[i] = math.Exp(xs[i])
        sum += ys[i]
    }

        for i: = range ys {
        ys[i] /= sum
    }

        return ys
}
