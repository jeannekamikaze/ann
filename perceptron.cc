#include <vector>
#include <functional>
#include <random>
#include <iostream>

std::ostream& operator<< (std::ostream& os, const std::vector<double>& vector)
{
    if (vector.size() == 0)
    {
        os << "[]";
    }
    else
    {
        os << "[";
        for (std::size_t i = 0; i < vector.size()-1; ++i)
            os << vector[i] << ", ";
        os << vector[vector.size()-1];
        os << "]";
    }
    return os;
}

using RandGen = std::mt19937;
using Activation_Function = std::function<double(double)>;

/// Evaluate the perceptron.
double feedforward (
    const std::vector<double>& input,
    const std::vector<double>& weight,
    const Activation_Function& activate)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        sum += input[i] * weight[i];
    }
    return activate(sum);
}

/// Binary activation function.
double activate (double out)
{
    return out >= 0.0 ? 1.0 : -1.0;
}

/// Adjust the perceptron's weights.
void adjust_weights (
    double c, // learning rate
    double error,
    const std::vector<double>& input,
    std::vector<double>& weight)
{
    for (std::size_t i = 0; i < weight.size(); ++i)
    {
        weight[i] = weight[i] + c * error * input[i];
    }
}

/// Train a perceptron so that it is able to separate points above and below
/// the y = x line.
void train_yx (RandGen& gen, std::vector<double>& weight)
{
    std::uniform_real_distribution<double> rand(-10, 10);
    const int max_iterations = 10000;
    double c = 0.1; // learning rate

    std::vector<double> input(3);
    input[2] = 1; // bias is always 1

    for (int i = 0; i < max_iterations; ++i)
    {
        // Generate an input with a known solution.
        input[0]  = rand(gen); // x
        input[1]  = rand(gen); // y
        double sol = input[1] >= input[0] ? 1.0 : -1.0; // +1 above the line, -1 below

        // Adjust the perceptron's weights.
        double out = feedforward(input, weight, activate);
        double error = sol - out;
        adjust_weights(c, error, input, weight);

        // Adjust learning rate to help convergence
        c = 0.999 * c;

        std::cout << "W: " << weight << ", E: " << error << ", C: " << c << std::endl;
    }
}

/// Return a vector of size 'n' with values uniformly distributed in [a,b).
template <typename RandGen>
std::vector<double> random_vector (RandGen& gen, std::size_t n, double a, double b)
{
    std::vector<double> vector(n);
    std::uniform_real_distribution<double> rand(a,b);
    for (double& x : vector)
    {
        x = rand(gen);
    }
    return vector;
}

int main ()
{
    std::random_device rd;
    RandGen gen(rd());

    std::vector<double> weight = random_vector(gen, 3, -1, 1);
    train_yx(gen, weight);

    std::vector<std::vector<double>> inputs = {
        // last value is the bias
        { 2, 5, 1 },
        { 3, 1, 1 }
    };

    for (auto input : inputs)
    {
        double output = feedforward(input, weight, activate);
        std::cout << input << " -> " << output << std::endl;
    }

    return 0;
}
