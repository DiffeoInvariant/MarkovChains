
#include<vector>
#include"Eigen/Core"
#include"Eigen/Eigenvalues"
#include "MatrixFunctions.hpp"
#include "Eigen/src/Core/util/Constants.h"
#include"Eigen/Dense"
#include"MarkovChain.hpp"
#include"Distributions.hpp"
#include<random>
#include<iostream>
#include<cmath>
#include<cstring>
#include<type_traits>
#include<limits>
#include<boost/foreach.hpp>
#include "HMMBase.hpp"

//default constructor
template<> HMMBase::HMMBase(){
    min_variance = 1.0e-5;
    n_distribution = 0;
}
//parametrized constructor
template<typename _dtype, typename dist, bool independent = false> HMMBase::HMMBase(vector<_dtype> _data, int num_distributions){
    data = _data;
    n_distribution = num_distributions;
}

/**
 * @author: Emily Jakobs
 * @return: are all variances \geq the minimum variance?
 */
template<typename _dtype, typename dist, bool independent = false> bool HMMBase::has_acceptable_variance(){
    for(int i = 0; i < distributions.size(); i++){
        if(distributions[i].variance < min_variance){
            return false;
        }
    }
    return true;
}

/**
 * @author: Emily Jakobs
 * @return: likelihood function of discrete random variable given our data and the model's parameters
 */
template<>
double HMMBase::discrete_likelihood(){
    if(!has_acceptable_variance()){
        throw "Error: At least one variance is too small, and could lead to an unbounded likelihood";
    }
    int n = data.size();//how many datapoints?
    double prod = 1.0;
    for(int i = 0; i < n; i ++){
        prod *= discrete_weighted_prob(data[i]);
    }
    return prod;
}

/**
 * @author: Emily Jakobs
 * @return: likelihood function of continuous random variable given our data and the model's parameters
 */
template<>
double HMMBase::continuous_likelihood(){
    if(!has_acceptable_variance()){
        throw "Error: At least one variance is too small, and could lead to an unbounded likelihood";
    }
    int n = data.size();
    double prod = 1.0;
    for(int i = 0; i < n - 1; i++){
        prod *= continuous_weighted_prob(data[i], data[i+1]);
    }
    return prod;
}
