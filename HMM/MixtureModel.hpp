#ifndef MixtureModel_hpp
#define MixtureModel_hpp
#include<vector>
#include"Eigen/Core"
#include"Eigen/Eigenvalues"
#include "MatrixFunctions.hpp"
#include "Eigen/src/Core/util/Constants.h"
#include"Eigen/Dense"
#include"Distributions.hpp"
#include<random>
#include<iostream>
#include<cmath>
#include<cstring>
#include<type_traits>
#include<limits>

using namespace std;

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
     * @param dist: what sort of distribution? Must choose from options in Markov namespace (Distributinos.hpp)
     * @param independent: are the distributions independent?
     * @summary: template to initialize a chosen solver class
     */
    template<typename _dtype, typename dist, bool independent = false> class MixtureModel
    {
    protected:
        
        double min_variance; //minimum allowable variance to avoid unbounded likelihood
        
        vector<_dtype> data;
        
        int n_distribution;
        //distribution parameter vectors
        vector<distribution_parameter> theta;
        //mixing parameters
        vector<double>delta;
        
        //vector of distributions
        vector<dist> distributions;
        
    public:
        
        MixtureModel() {};
        //true iff all variances are greater than or equal to min_variance
        bool has_acceptable_variance(){
            for(int i = 0; i < distributions.size(); i++){
                if(distributions[i].variance < min_variance){
                    return false;
                }
            }
            return true;
        }
        
        void set_theta(vector<distribution_parameter> &_tht);
        
        void forward_parameter_transform();
        
        void inverse_parameter_transform();
    
        double discrete_weighted_prob(_dtype x);
        
        double continuous_weighted_prob(_dtype x, _dtype y);
        //discrete likelihood function
        double discrete_likelihood(){
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
        
        double continuous_likelihood(){
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
        
        void set_distribution_params();
        
    
    };
    //for independent
    
    /**
     *@author: Zane Jakobs
     * @summary: transform into working parameters for independent Poisson mixture model
     */
    template<> void MixtureModel<int, Poisson, true>::forward_parameter_transform(){

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
            sum += delta[j];
        }
        for(int i = 0; i < delta.size(); i ++){
            delta[i] = log(delta[i] / (1-sum) );
        }
    }
    //get the original params back
    template<> void MixtureModel<int, Poisson, true>::inverse_parameter_transform(){
    
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
                sum += pow(e,delta[j]);
        }
        for(int i = 0; i < delta.size(); i++){
            delta[i] = pow(e,delta[i]) / (1 + sum) ;
        }
    }
    //set theta with a vector of params, since that works for Poisson
    template<> void MixtureModel<int, Poisson, true>::set_theta(vector<distribution_parameter>& tht){
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
    
    template<> void MixtureModel<int, Poisson, true>::set_distribution_params(){
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
    
    //weighted probability of getting x
    template<>
    double MixtureModel<int, Poisson, true>::discrete_weighted_prob(int x){
        if(theta.size() != n_distribution){
            throw "Error: parameter and distribution size mismatch.";
        }
        if(delta.size() != n_distribution ){
            throw "Error: distribution weight size and distribution size mismatch.";
        }
        
        double sum = 0;
        for(int i = 0; i < n_distribution; i++){
            sum += delta[i]*double(distributions[i].P_eq(x)); //sum += delta_i * P(X_i = x)
        }
        return sum;
    }
    

   
    typedef MixtureModel<int, Poisson, true>  IndependentPoissonMixtureModel ;
}

#endif /* MixtureModel_hpp */
