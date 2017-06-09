// This is an implementation for the multiple-neuron perceptron example
// in chapter 4 of the Neural Network Design book by Martin Hagan.

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

using namespace Eigen;

double hardlim (double x)
{
    return x >= 0 ? 1 : 0;
}

double feedforward (const MatrixXd& W, const VectorXd& p)
{
    return hardlim((W*p)[0]);
}

int main ()
{
    // Set initial values for the weight matrix.
    double bias = 0.5;
    MatrixXd W(1,4); W << 0.5, -1.0, -0.5, bias;

    // Example input/output pairs.
    // Extended with a last value of 1 that multiplies the bias.
    VectorXd p1(4); p1 << 1, -1, -1, 1;
    VectorXd p2(4); p2 << 1, +1, -1, 1;
    std::vector<VectorXd> inputs { p1, p2 };
    std::vector<double> targets { 0, 1 };

    // Train the network.
    // It is kind of dumb to iterate many times here because we know
    // the network converges in 3 steps for this particular problem,
    // but this is what you would do in the general case.
    for (int i = 0; i < 100; ++i)
    {
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            const VectorXd& p = inputs[i];
            double t = targets[i];
            double o = feedforward(W,p); // network output
            double e = t-o; // error
            W = W + (e*p).transpose(); // update weights using perceptron learning rule
        }
    }

    std::cout << "Weight matrix: " << W << std::endl;
    std::cout << p1.transpose() << " -> " << feedforward(W,p1) << std::endl;
    std::cout << p2.transpose() << " -> " << feedforward(W,p2) << std::endl;

    return 0;
}
