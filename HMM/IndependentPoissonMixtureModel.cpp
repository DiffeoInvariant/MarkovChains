
#include <stdio.h>
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
#include"HMMBase.hpp"
#include "IndependentPoissonMixtureModel.hpp"


/**
 *@author: Zane Jakobs
 * @summary: transform into working parameters for independent Poisson mixture model
 */
template<> void IndependentPoissonMixtureModel::forward_parameter_transform(){
    
    if(theta.size() != n_distribution){
        throw "Error: parameter and distribution size mismatch.";
    }
    if(delta.size() != n_distribution ){
        throw "Error: distribution weight size and distribution size mismatch.";
    }
    for(int i = 0; i < n_distribution; i++){
        theta[i].param[0] = log(theta[i].param[0]);
    }
    double sum = 0;
    for(int j = 0; j < n_distribution - 1; j++){
        sum += hidden_states[j].param(0);
    }
    for(int i = 0; i < delta.size(); i ++){
        hidden_states[i].param(0) = log(hidden_states[i].param(0) / (1-sum) );
    }
}

/**
 *@author: Zane Jakobs
 * @summary: transform from working parameters back into
 * regular parameters for independent Poisson mixture model. Functional inverse of
 * forward_parameter_transform().
 */
template<> void HMM<int, Poisson, true>::inverse_parameter_transform(){
    
    if(theta.size() != n_distribution){
        throw "Error: parameter and distribution size mismatch.";
    }
    if(delta.size() != n_distribution ){
        throw "Error: distribution weight size and distribution size mismatch.";
    }
    
    const double e = 2.718281828459045;
    for(int i = 0; i < n_distribution; i++){
        theta[i].param[0] = pow(e,theta[i].param[0]);
    }
    double sum = 0;
    for(int j = 0; j < n_distribution - 1; j++){
        sum += pow(e,hidden_states[j].param(0));
    }
    for(int i = 0; i < delta.size(); i++){
        hidden_states[i].param(0) = pow(e,hidden_states[i].param(0)) / (1 + sum) ;
    }
}
/**
 *@author: Zane Jakobs
 * @param x: probability of getting x from Poisson model
 */
template<>
double IndependentPoissonMixtureModel::discrete_weighted_prob(int x){
    if(theta.size() != n_distribution){
        throw "Error: parameter and distribution size mismatch.";
    }
    if(delta.size() != n_distribution ){
        throw "Error: distribution weight size and distribution size mismatch.";
    }
    
    double sum = 0;
    for(int i = 0; i < n_distribution; i++){
        sum += hidden_states[i].param(0)*double(distributions[i].P_eq(x)); //sum += delta_i * P(X_i = x)
    }
    return sum;
}

/**
 *@author: Zane Jakobs
 *@return: expected value of the model given params
 */
template<>
double IndependentPoissonMixtureModel::model_expectation(){
    double sum = 0;
    for(int i = 0; i < n_distribution; i++){
        sum += hidden_states[i].param(0) * theta[i].param(0);
    }
    return sum;
}

/**
 *@author: Zane Jakobs
 *@return: Variance of the model given params
 */
template<>
double IndependentPoissonMixtureModel::model_variance(){
    double sum = 0;
    //E(X^2) = sum_i delta_i (lambda_i + lambda_i^2)
    for(int i = 0; i < n_distribution; i++){
        double lambda = theta[i].param(0);
        sum += hidden_states[i].param(0) * ( lambda + lambda*lambda);
    }
    
    double mu = model_expectation();
    sum -= mu * mu;
    
    return sum;
}

/**
 *@author: Zane Jakobs
 * @summary: sets lambda_i to DBL_MAX
 */
template<> void IndependentPoissonMixtureModel::set_default_distribution_params(){
    if(distributions.size() != n_distribution){
        distributions.resize(n_distribution);
    }
    if(theta.size() != n_distribution){
        vector<distribution_parameter> t;
        t.resize(n_distribution);//fill t with default double max val for params
        for(int i = 0; i < n_distribution; i ++){
            t[i].param[0] = DBL_MAX;
        }
        set_theta(t);
    }
    for(int i = 0; i < n_distribution; i++){
        distributions[i].setLambda(theta[i].param[0]);
    }
}

/**
 *@author: Zane Jakobs
 * @param tht: parameter vector
 */
template<> void IndependentPoissonMixtureModel::set_theta(vector<distribution_parameter>& tht){
    if(! f_or_d(tht)){
        throw "Error: distribution parameter for Poisson distribution not float or double";
    }
    if(tht.size() != n_distribution){
        throw "Error: input parameter vector size does not match number of distributions";
    }
    const int paramsPerDist = 1;
    if(theta.size() != n_distribution){
        theta.resize(n_distribution);
    }
    for(int i = 0; i < n_distribution; i++){
        theta[i].param.resize(paramsPerDist);//set param
        theta[i] = tht[i];
    }
}
