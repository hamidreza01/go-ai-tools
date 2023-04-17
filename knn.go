package main

import (
    "fmt"
    "math"
    "sort"
)

type DataPoint struct {
    Features []float64
    Label    float64
}

type Distance struct {
    Index int
    Value float64
}

type KNN struct {
    K          int
    DataPoints []DataPoint
}

func NewKNN(k int, dataPoints []DataPoint) *KNN {
    return &KNN{
        K:          k,
        DataPoints: dataPoints,
    }
}

func (knn *KNN) Distance(point1 []float64, point2 []float64) float64 {
    sum := 0.0
    for i := 0; i < len(point1); i++ {
        sum += math.Pow(point1[i]-point2[i], 2)
    }
    return math.Sqrt(sum)
}

func (knn *KNN) Classify(point []float64) float64 {
    distances := make([]Distance, len(knn.DataPoints))
    for i := 0; i < len(knn.DataPoints); i++ {
        distances[i] = Distance{
            Index: i,
            Value: knn.Distance(point, knn.DataPoints[i].Features),
        }
    }
    sort.Slice(distances, func(i, j int) bool { return distances[i].Value < distances[j].Value })

    count := make(map[float64]int)
    for i := 0; i < knn.K; i++ {
        count[knn.DataPoints[distances[i].Index].Label]++
    }

    maxCount := 0
    maxLabel := 0.0
    for label, c := range count {
        if c > maxCount {
            maxCount = c
            maxLabel = label
        }
    }

    return maxLabel
}
