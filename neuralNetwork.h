//
// Created by TristanXia on 2019/7/20.
//
#include "Neuron.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H


class neuralNetwork {
private:
    Neuron h1, h2, o1;

    double randomNormal(){
        std::default_random_engine e;
        std::normal_distribution<double> n;
        return n(e);
    }

    double sigmoid(double x){
        return 1 / (1 + std::exp(-x));
    }

    double deriv_Sigmoid(double x){
        double fx = sigmoid(x);
        return fx * (1 - fx);
    }

    double mse_loss(std::vector<double> y_true, std::vector<double> y_pred){
        double sum(0);
        for(uint32_t i = 0; i < y_true.size(); ++i){
            sum += std::pow(y_true[i] - y_pred[i], 2);
        }
        return sum / y_true.size();
    }

public:
    neuralNetwork();

    double feedForward(std::vector<double>& x);

    void train(std::vector<std::vector<double>> data, std::vector<double> all_y_trues);

    void printResult(std::vector<double> subject, std::string name);
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
