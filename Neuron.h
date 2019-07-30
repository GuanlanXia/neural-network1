//
// Created by TristanXia on 2019/7/20.
//
#include <vector>
#include <cmath>
#ifndef NEURAL_NETWORK_NEURON_H
#define NEURAL_NETWORK_NEURON_H



class Neuron {
private:
    std::vector<double> weights;
    double bias;
    double output;

    double mSigmoid(double x){
        return 1 / (1 + std::exp(-x));
    }
     friend class neuralNetwork;
public:
    Neuron();

    Neuron(std::vector<double>& mWeight, double mBias);

    double sum(std::vector<double>& x);

    double feedForward(std::vector<double>& inputs);
};


#endif //NEURAL_NETWORK_NEURON_H
