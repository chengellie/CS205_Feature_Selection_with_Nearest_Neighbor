#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::vector<std::vector<double>> parseFile(const std::string &filename) {
    std::vector<std::vector<double>> data;
    std::ifstream fileIn(filename);
    if (!fileIn.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    double num;

    std::vector<double> row;
    while (getline(fileIn, line)) {
        std::istringstream ssin(line);
        while (ssin >> num) {
            row.push_back(num);
        }
        data.push_back(row);
        row.clear();
    }

    fileIn.close();
    return data;
}

void printData(std::vector<std::vector<double>> &data) {
    for (unsigned i = 0; i < data.size(); i++) {
        for (unsigned j = 0; j < data.at(i).size(); j++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

double euclideanDistance(int instanceIdx1, int instanceIdx2,
                         std::vector<std::vector<double>> &data,
                         std::vector<int> &features,
                         std::unordered_map<std::string, double> &calculated) {
    double squaredSums = 0;
    for (unsigned i = 0; i < features.size(); i++) {
        // std::string key = std::to_string(i) + "-";
        // key += instanceIdx1 <= instanceIdx2
        //            ? std::to_string(instanceIdx1) + "-" +
        //                  std::to_string(instanceIdx2)
        //            : std::to_string(instanceIdx2) + "-" +
        //                  std::to_string(instanceIdx1);
        // if (calculated.find(key) != calculated.end()) {
        // std::cout << "found " << key << " " << calculated[key] <<
        // std::endl;
        // squaredSums += calculated[key];
        // } else {
        double diff = (data[instanceIdx1][features.at(i)] -
                       data[instanceIdx2][features.at(i)]);
        double squared = diff * diff;
        squaredSums += squared;
        // calculated[key] = squared;
        // std::cout << "insert " << key << " " << squared << std::endl;
        // }
    }
    return std::sqrt(squaredSums);
    // return squaredSums;
}

bool nearestNeighbor(int instanceIdx, std::vector<std::vector<double>> &data,
                     std::vector<int> &features,
                     std::vector<std::pair<int, int>> &trainingRanges,
                     std::unordered_map<std::string, double> &calculated,
                     bool verbose = false) {
    double nnDist = std::numeric_limits<double>::max();
    int nnIdx = -1;

    for (unsigned i = 0; i < trainingRanges.size(); i++) {
        for (unsigned j = trainingRanges.at(i).first;
             j <= trainingRanges.at(i).second; j++) {
            // if (j != instanceIdx) {
            double currDist =
                euclideanDistance(instanceIdx, j, data, features, calculated);
            if (currDist < nnDist) {
                nnDist = currDist;
                nnIdx = j;
            }
            // }
        }
    }

    if (verbose) {
        std::cout << "Object " << instanceIdx + 1 << " nearest neighbor: "
                  << "object " << nnIdx + 1 << std::endl;
        std::cout << "\tEuclidean distance: " << nnDist << ", Object "
                  << instanceIdx + 1 << " class: " << data[instanceIdx][0]
                  << ", Object " << nnIdx + 1 << " class: " << data[nnIdx][0]
                  << std::endl;
    }

    return data[nnIdx][0] == data[instanceIdx][0];
}

bool nearestNeighbor(int instanceIdx, std::vector<std::vector<double>> &data,
                     std::vector<int> &features,
                     std::unordered_map<std::string, double> &calculated,
                     bool verbose = false) {
    if (instanceIdx > data.size() - 1) {
        throw std::out_of_range("Instance index out of bounds");
    }

    int beforeRangeBegin = 0;
    int beforeRangeEnd = instanceIdx - 1;
    int afterRangeBegin = instanceIdx + 1;
    int afterRangeEnd = data.size() - 1;
    std::vector<std::pair<int, int>> trainingRanges;
    if (beforeRangeEnd >= beforeRangeBegin) {
        trainingRanges.push_back({beforeRangeBegin, beforeRangeEnd});
    }
    if (afterRangeBegin <= afterRangeEnd) {
        trainingRanges.push_back({afterRangeBegin, afterRangeEnd});
    }

    return nearestNeighbor(instanceIdx, data, features, trainingRanges,
                           calculated, verbose);
}

double kFoldCrossValidation(std::vector<std::vector<double>> &data,
                            std::vector<int> &features,
                            std::unordered_map<std::string, double> &calculated,
                            int kFold = -1) {
    int correctClassif = 0;

    if (kFold == -1) {
        for (unsigned i = 0; i < data.size(); i++) {
            if (nearestNeighbor(i, data, features, calculated)) {
                correctClassif++;
            }
        }
    }
    // else {
    //     for (unsigned i = 0; i < kFold; i++) {
    //         for (unsigned j = i * kFold; j < (i + 1) * kFold; j++) {
    //             if (nearestNeighbor(j, data,)) {
    //                 correctClassif++;
    //             }
    //         }
    //     }
    // }

    double accuracy = (double)correctClassif / data.size();
    return accuracy;
}

std::string getFeatureSetStr(std::vector<int> &features) {
    std::string ret = "{ ";
    for (unsigned i = 0; i < features.size(); i++) {
        ret += std::to_string(features.at(i)) + " ";
    }
    ret += "}";
    return ret;
}

std::vector<std::vector<double>> sampleData(
    std::vector<std::vector<double>> &data, double sampling) {
    int numInstances = floor(sampling * data.size());

    std::vector<std::vector<double>> sampled(numInstances);
    for (unsigned i = 0; i < sampled.size(); i++) {
        sampled.at(i) = data.at(i);
    }
    return sampled;
}

void featureSearchForwardSelection(std::vector<std::vector<double>> &data,
                                   double sampling = 1) {
    std::cout << "Feature Search with Forward Selection" << std::endl;

    clock_t start = clock();
    std::vector<int> featureSet;
    std::unordered_set<int> visited;
    std::unordered_map<std::string, double> calculated;

    double bestOverallAcc = 0;
    std::vector<int> bestFeatureSet;

    for (unsigned i = 1; i < data[0].size(); i++) {
        std::cout << "Considering " << i << "-feature sets" << std::endl;
        double bestSoFarAcc = 0;
        int bestFeature = 0;
        featureSet.push_back(-1);
        for (unsigned j = 1; j < data[0].size(); j++) {
            double currAcc = 0;
            if (visited.find(j) == visited.end()) {
                featureSet.at(i - 1) = j;
                currAcc = kFoldCrossValidation(data, featureSet, calculated);
                if (currAcc > bestSoFarAcc) {
                    bestSoFarAcc = currAcc;
                    bestFeature = j;
                }
                std::cout << "\tUsing feature(s) "
                          << getFeatureSetStr(featureSet) << " accuracy is "
                          << currAcc * 100 << "%" << std::endl;
            }
        }
        visited.insert(bestFeature);
        featureSet.at(i - 1) = bestFeature;
        if (bestSoFarAcc > bestOverallAcc) {
            bestOverallAcc = bestSoFarAcc;
            bestFeatureSet.push_back(bestFeature);
        }
        std::cout << std::endl
                  << "\tBest feature set is " << getFeatureSetStr(featureSet)
                  << " with accuracy " << bestSoFarAcc * 100 << "%"
                  << std::endl;
        std::cout << "\tElapsed runtime is "
                  << float(clock() - start) / CLOCKS_PER_SEC << " seconds\n\n";
    }
    std::cout << std::endl
              << "Best overall feature set is "
              << getFeatureSetStr(bestFeatureSet) << " with accuracy "
              << bestOverallAcc * 100 << "%" << std::endl;

    clock_t end = clock();
    std::cout << "Total runtime is " << float(end - start) / CLOCKS_PER_SEC
              << " seconds\n";
}

void featureSearchBackwardElimination(std::vector<std::vector<double>> &data,
                                      double sampling = 1) {
    std::cout << "Feature Search with Backward Elimination" << std::endl;

    clock_t start = clock();
    std::vector<int> featureSet;
    std::unordered_set<int> visited;
    std::unordered_map<std::string, double> calculated;

    double bestOverallAcc = 0;
    std::vector<int> bestFeatureSet;
    for (unsigned i = 1; i < data[0].size(); i++) {
        featureSet.push_back(i);
        bestFeatureSet.push_back(i);
    }

    for (unsigned i = 1; i < data[0].size() - 1; i++) {
        std::cout << "Considering " << data[0].size() - i - 1 << "-feature sets"
                  << std::endl;
        double bestSoFarAcc = 0;
        int bestFeature = 0;
        int bestFeatureIdx = -1;
        for (unsigned j = 0; j < featureSet.size(); j++) {
            double currAcc = 0;
            int currFeature = featureSet.at(0);
            // if (visited.find(currFeature) == visited.end()) {
            featureSet.erase(featureSet.begin());
            currAcc = kFoldCrossValidation(data, featureSet, calculated);
            if (currAcc > bestSoFarAcc) {
                bestSoFarAcc = currAcc;
                bestFeature = currFeature;
                bestFeatureIdx = j;
            }
            std::cout << "\tUsing feature(s) " << getFeatureSetStr(featureSet)
                      << " accuracy is " << currAcc * 100 << "% (removing "
                      << currFeature << ")" << std::endl;
            featureSet.push_back(currFeature);
            // }
        }
        // visited.insert(bestFeature);
        featureSet.erase(featureSet.begin() + (bestFeatureIdx));
        if (bestSoFarAcc > bestOverallAcc) {
            bestOverallAcc = bestSoFarAcc;
            bestFeatureSet = featureSet;
        }
        std::cout << std::endl
                  << "\tBest feature set is " << getFeatureSetStr(featureSet)
                  << " with accuracy " << bestSoFarAcc * 100 << "% (removing "
                  << bestFeature << ")" << std::endl;
        std::cout << "\tElapsed runtime is "
                  << float(clock() - start) / CLOCKS_PER_SEC << " seconds\n\n";
    }
    std::cout << std::endl
              << "Best overall feature set is "
              << getFeatureSetStr(bestFeatureSet) << " with accuracy "
              << bestOverallAcc * 100 << "%" << std::endl;

    clock_t end = clock();
    std::cout << "Total runtime is " << float(end - start) / CLOCKS_PER_SEC
              << " seconds\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    std::string filename = "CS170_large_Data__21.txt";
    // std::string filename = "test.txt";
    std::vector<std::vector<double>> data = parseFile(filename);

    std::cout << "Dataset " << filename << " has " << data[0].size() - 1
              << " features and " << data.size() << " instances\n";

    double samplingRate = 1;
    if (samplingRate < 1) {
        data = sampleData(data, samplingRate);
        std::cout << "Applied sampling rate of " << samplingRate
                  << ", sampled dataset has " << data.size() << " instances\n";
    }

    std::cout << "\n";
    featureSearchForwardSelection(data);

    // featureSearchBackwardElimination(data);
}