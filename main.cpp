#include <iostream>
#include "Neuron.h"
#include "neuralNetwork.h"
int main() {
    std::vector<std::vector<double>> data{{-2, -1}, {25, 6}, {17, 4}, {-15, -6}};
    std::vector<double> all_y_trues{1, 0, 0, 1};
    neuralNetwork network;
    std::vector<double> emily{-7, -3};
    std::vector<double> frank{20, 2};
    std::cout << "Untrained prediction: " << std::endl;
    network.printResult(emily, "Emily");
    network.printResult(frank, "Frank");
    std::cout << "Begin training..." << std::endl;
    network.train(data, all_y_trues);
    std::cout << "Training completed." << std::endl;
    network.printResult(emily, "Emily");
    network.printResult(frank, "Frank");
    return 0;
}