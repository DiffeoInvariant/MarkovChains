

#ifndef IndependentPoissonMixtureModel_hpp
#define IndependentPoissonMixtureModel_hpp

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
using namespace std;

namespace Markov {
    template<>
    class IndependentPoissonMixtureModel : HMMBase<int, Poisson, true> {
        
        //default constructor
        IndependentPoissonMixtureModel();
        
        /**
         *@author: Emily Jakobs
         * @summary: transform into working parameters for independent Poisson mixture model
         */
        void forward_parameter_transform();
        
    
        /**
        *@author: Emily Jakobs
        * @summary: transform from working parameters back into
        * regular parameters for independent Poisson mixture model. Functional inverse of
        * forward_parameter_transform().
        */
        void inverse_parameter_transform();
    
        /**
         *@author: Emily Jakobs
         * @param tht: parameter vector
         */
        void set_theta(vector<distribution_parameter>& tht);
    
        /**
        *@author: Emily Jakobs
         * @summary: sets lambda_i to DBL_MAX
         */
        void set_default_distribution_params();
    
        /**
         *@author: Emily Jakobs
         * @param x: probability of getting x from Poisson model
         */
        double discrete_weighted_prob(int x);
        
        /**
         *@author: Emily Jakobs
         *@return: expected value of the model given params
         */
        double model_expectation();
        
        /**
         *@author: Emily Jakobs
         *@return: Variance of the model given params
         */
        double model_variance();
        
        /**
         *@author: Emily Jakobs
         * @param hiddenState, observed_state: what state is the model in?
         * @return: prediction of the next step (maybe distribution? figure this out later)
         */
        int predict(int hidden_state, int observed_state);
        
    };

#endif /* IndependentPoissonMixtureModel_hpp */
