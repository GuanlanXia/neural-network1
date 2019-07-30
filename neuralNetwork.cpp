//
// Created by TristanXia on 2019/7/20.
//
#include <iostream>
#include "neuralNetwork.h"
neuralNetwork::neuralNetwork() {
    std::vector<double> weights1{randomNormal(), randomNormal()};
    std::vector<double> weights2{randomNormal(), randomNormal()};
    std::vector<double> weights3{randomNormal(), randomNormal()};
    h1.weights = weights1;
    h1.bias = randomNormal();
    h2.weights = weights2;
    h2.bias = randomNormal();
    o1.weights = weights3;
    o1.bias = randomNormal();
}

double neuralNetwork::feedForward(std::vector<double> &x) {
    h1.output = h1.feedForward(x);
    h2.output = h2.feedForward(x);
    std::vector<double> tmp{h1.output, h2.output};
    o1.output = o1.feedForward(tmp);
    return o1.output;
}

void neuralNetwork::train(std::vector<std::vector<double>> data, std::vector<double> all_y_trues){
    double learn_rate = 0.1;
    uint32_t epochs = 1000;
    for(uint32_t epoch = 0; epoch < epochs; ++epoch){
        for(int i = 0; i < data.size(); ++i){
            double y_pred = feedForward(data[i]);
            std::vector<double> tmp{h1.output, h2.output};
            //calculate partial derivatives
            double d_L_d_ypred = -2 * (all_y_trues[i] - y_pred);
            //neuron o1
            double o1_sumDeriv = deriv_Sigmoid(o1.sum(tmp));
            double d_ypred_d_w5 = h1.output * o1_sumDeriv;
            double d_ypred_d_w6 = h2.output * o1_sumDeriv;
            double d_ypred_d_b3 = o1_sumDeriv;
            double d_ypred_d_h1 = o1.weights[0] * o1_sumDeriv;
            double d_ypred_d_h2 = o1.weights[1] * o1_sumDeriv;
            //neuron h1
            double sum_h1 = h1.sum(data[i]);
            double d_h1_d_w1 = data[i].at(0) * deriv_Sigmoid(sum_h1);
            double d_h1_d_w2 = data[i].at(1) * deriv_Sigmoid(sum_h1);
            double d_h1_d_b1 = deriv_Sigmoid(sum_h1);
            //neuron h2
            double sum_h2 = h2.sum(data[i]);
            double d_h2_d_w3 = data[i].at(0) * deriv_Sigmoid(sum_h2);
            double d_h2_d_w4 = data[i].at(1) * deriv_Sigmoid(sum_h2);
            double d_h2_d_b2 = deriv_Sigmoid(sum_h2);
            //update weights and biases
            //neuron h1
            h1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
            h1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
            h1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;
            //neuron h2
            h2.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
            h2.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
            h2.bias -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;
            //neuron o1
            o1.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
            o1.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
            o1.bias -= learn_rate * d_L_d_ypred * d_ypred_d_b3;
        }
        if(epoch % 100 == 0){
            std::vector<double> yPreds;
            for(int i = 0; i < data.size(); ++i){
                yPreds.push_back(feedForward(data[i]));
            }
            double loss = mse_loss(all_y_trues, yPreds);
            std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
        }
    }
}

void neuralNetwork::printResult(std::vector<double> subject, std::string name) {
    double yPred = feedForward(subject);
    std::cout << name << ": " << yPred;
    if(yPred < 0.1){
        std::cout << " Male" << std::endl;
    } else if(yPred > 0.9){
        std::cout << " Female" << std::endl;
    } else{
        std::cout << " ambiguous" << std::endl;
    }
}