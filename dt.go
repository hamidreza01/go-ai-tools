package main

import (
    "fmt"
    "math"
)

type DecisionTree struct {
    root * node
}

type node struct {
    splitFeature int
    splitValue float64
    left * node
    right * node
    label int
}

type DataPoint struct {
    features[] float64
    label int
}

func NewDecisionTree(dataPoints[] DataPoint) * DecisionTree {
    root: = buildTree(dataPoints, 0)
    return &DecisionTree {
        root: root
    }
}

func(dt * DecisionTree) classify(features[] float64) int {
    return traverseTree(features, dt.root)
}

func buildTree(dataPoints[] DataPoint, depth int) * node {
    if len(dataPoints) == 0 {
        return &node {
            label: -1
        }
    }

    label: = dataPoints[0].label
    sameLabel: = true
    for _, dp: = range dataPoints {
        if dp.label != label {
            sameLabel = false
            break
        }
    }
    if sameLabel {
        return &node {
            label: label
        }
    }

    splitFeature: = -1
    maxInfoGain: = -math.MaxFloat64
    for i: = 0;
    i < len(dataPoints[0].features);
    i++{
        infoGain: = computeInfoGain(dataPoints, i)
        if infoGain > maxInfoGain {
            maxInfoGain = infoGain
            splitFeature = i
        }
    }

    leftDataPoints: = [] DataPoint {}
    rightDataPoints: = [] DataPoint {}
    splitValue: = computeSplitValue(dataPoints, splitFeature)
    for _, dp: = range dataPoints {
        if dp.features[splitFeature] < splitValue {
            leftDataPoints = append(leftDataPoints, dp)
        } else {
            rightDataPoints = append(rightDataPoints, dp)
        }
    }

    left: = buildTree(leftDataPoints, depth + 1)
    right: = buildTree(rightDataPoints, depth + 1)

    return &node {
        splitFeature: splitFeature,
        splitValue: splitValue,
        left: left,
        right: right,
    }
}

func traverseTree(features[] float64, n * node) int {
    if n.label != -1 {
        return n.label
    }
    if features[n.splitFeature] < n.splitValue {
        return traverseTree(features, n.left)
    } else {
        return traverseTree(features, n.right)
    }
}

func computeInfoGain(dataPoints[] DataPoint, featureIndex int) float64 {
    totalEntropy: = computeEntropy(dataPoints)
    leftDataPoints: = [] DataPoint {}
    rightDataPoints: = [] DataPoint {}
    splitValue: = computeSplitValue(dataPoints, featureIndex)
    for _,
    dp: = range dataPoints {
        if dp.features[featureIndex] < splitValue {
            leftDataPoints = append(leftDataPoints, dp)
        } else {
            rightDataPoints = append(rightDataPoints, dp)
        }
    }
    leftEntropy: = computeEntropy(leftDataPoints)
    rightEntropy: = computeEntropy(rightDataPoints)
    leftFraction: = float64(len(leftDataPoints)) / float64(len(dataPoints))
    rightFraction: = float64(len(rightDataPoints)) / float64(len(dataPoints))
    return totalEntropy - leftFraction * leftEntropy - rightFraction * rightEntropy
}

func computeEntropy(dataPoints[] DataPoint) float64 {
    labelCounts: = make(map[int] int)
    for _,
    dp: = range dataPoints {
        labelCounts[dp.label]++
    }
    entropy: = 0.0
    for _,
    count: = range labelCounts {
        prob: = float64(count) / float64(len(dataPoints))
        entropy -= prob * math.Log2(prob)
    }
    return entropy
}

func computeSplitValue(dataPoints[] DataPoint, featureIndex int) float64 {
    sum: = 0.0
    for _,
    dp: = range dataPoints {
        sum += dp.features[featureIndex]
    }
    return sum / float64(len(dataPoints))
}
