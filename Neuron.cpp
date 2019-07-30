//
// Created by TristanXia on 2019/7/20.
//
#include <numeric>
#include "Neuron.h"


Neuron::Neuron(): weights({0, 0}), bias(0), output(0) {}

Neuron::Neuron(std::vector<double>& mWeight, double mBias): weights(mWeight), bias(mBias), output(0) {
}

double Neuron::sum(std::vector<double> &x) {
    return std::inner_product(weights.begin(), weights.end(), x.begin(), 0.0) + bias;
}

double Neuron::feedForward(std::vector<double> &inputs) {
    return mSigmoid(sum(inputs));
}