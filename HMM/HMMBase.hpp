#ifndef HMMBase_hpp
#define HMMBase_hpp
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

using namespace std;


/**
 \remark All functions and classes here are based off information in
 "Hidden Markov Models for Time Series: An Introduction Using R" by Zucchini, MacDonald, and Langrock
 
 */
namespace Markov {

    typedef struct distribution_parameter
    {
        //vector of parameters of a distribution
        vector<double> param;
    } distribution_parameter;
    
    template<typename...>
    using void_t = void;
    
    template<typename, typename = void>
    struct has_n_param : false_type {};
    //does the type have public variable n_param? If not, we can't use it as a distribution
    template<typename T>
    struct has_n_param<T, void_t<decltype( declval<T>().n_param)>> : true_type {};
    
    template<typename T>
    bool f_or_d( T &t1){
        return ( (is_same<T,float>::value) || (is_same<T,double>::value)); //true iff t1 is float or double
    }
    
    
    /**
     * @author: Zane Jakobss
     * @param _dtype: what sort of data will the model take?
     * @param n_distribution: how many distributions?
     * @param dist: what sort of distribution? Must choose from options in Markov namespace (Distributions.hpp)
     * @param independent: are the distributions independent?
     * @summary: template to initialize a chosen solver class
     */
    template<typename _dtype, typename dist, bool independent = false> class HMMBase
    {
    protected:
        
        double min_variance; //minimum allowable variance to avoid unbounded likelihood
        
        vector<_dtype> data;
        //number of hidden states
        int n_distribution;
        //distribution parameter vectors
        vector<distribution_parameter> theta;
        //vector of distributions
        vector<dist> distributions;
        //Markov chain for hidden states
        MarkovChain hidden_chain;
        //
        vector<distribution_parameter> hidden_states;
        
    public:
        
        HMMBase();
        //true iff all variances are greater than or equal to min_variance
        bool has_acceptable_variance();
        /**
         * @author: Zane Jakobs
         * @param _data: dataset to set data to
         */
        void setData(vector<_dtype> _data){
            data = _data;
        }
        
        void set_theta(vector<distribution_parameter> &_tht);
        
        double discrete_weighted_prob(_dtype x);
        
        double continuous_weighted_prob(_dtype x, _dtype y);
        //discrete likelihood function
        double discrete_likelihood;
        
        double continuous_likelihood;
        
        double discrete_state_dependent_probability(int hiddenState, _dtype x);
            
        
        void set_default_distribution_params();
        
        /**
         *@author: Zane Jakobs
         * @param hiddenState, observed_state: what state is the model in?
         * @return: prediction of the next step (maybe distribution? figure this out later)
         */
        _dtype predict(int hiddenState, _dtype observed_state);
        
    };
    
#endif /* HMMBase_hpp */
