#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <cfloat>
#include <algorithm>

#include "common.cuh"

auto read_input(std::string& name, std::string& comment, std::string& type,
                int& dimension, std::string& edge_weight_fromat)
{
    std::string line;
    std::getline(std::cin, line);
    name = line.substr(7);
    std::getline(std::cin, line);
    comment = line.substr(10);
    std::getline(std::cin, line);
    type = line.substr(7);
    std::getline(std::cin, line);
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    dimension = std::stoi(line.substr(10));
    std::getline(std::cin, line);
    edge_weight_fromat = line.substr(19);
    std::getline(std::cin, line); // read separator line

    // std::cout<< "name: " << name << std::endl;
    // std::cout<< "comment: " << comment << std::endl;
    // std::cout<< "type: " << type << std::endl;
    // std::cout<< "edge_weight_format: " << edge_weight_fromat << std::endl;
    // std::cout<< "dimension: " << dimension << std::endl;
    // std::cout<< "----------------------------------------" << std::endl;

    std::vector<double> x(dimension);
    std::vector<double> y(dimension);
    for (int i = 0; i < dimension; i++) {
        int ind; // read but not used
        std::cin >> ind >> x[i] >> y[i];
    }

    return std::make_pair(x, y);
}

void populate_dist(std::vector<float>& dist, const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist[cord(i, j)] = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2));
            dist[cord(i, j)] = std::max(dist[cord(i, j)], 1e-4f);
        }
        // hack: 1 / dist[cord(i, i)] != nan
        // dist[cord(i, i)] = 1;
    }
}