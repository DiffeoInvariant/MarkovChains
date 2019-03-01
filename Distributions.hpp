
#ifndef Distributions_hpp
#define Distributions_hpp

#include <stdio.h>
#include<random>
#include<cmath>
#include<limits>

using namespace std;

namespace Markov {
    
    int factorial(int n){
        if( n < 0){
            throw "Error: n < 0. Enter positive n or use Gamma function";
            return -1;
        }
        else if( n == 0 || n == 1){
            return 1;
        }
        else{
            return factorial(n-1);
        }
    }
    class Poisson
    {
    protected:
        double lambda; //lambda  = rate
        
    public:
        const int n_param = 1; //1 parameter for this class
        //default to largest possible lambda if no argument is passed
        Poisson(){
            lambda = DBL_MAX;
        }
        
        Poisson(double _lambda){
            lambda = _lambda;
        }
        
        void setLambda(double _lambda){
            lambda = _lambda;
        }
        
        double mean(){
            return lambda;
        }
        
        double variance(){
            return lambda;
        }
        //let X ~ \Lambda(k). This function returns P(X = x)
        double P_eq(int x){
            const double e = 2.71828182845904;
            double p = pow(e, -lambda)*pow(lambda,double(x))/factorial(int(x));
            return p;
        }
        //returns P( X <= x)
        double P_leq(int x){
            double sum = 0;
            for(int i = 0; i <= x; i++){
                sum += P_eq(i);
            }
            return sum;
        }
        //generates a sample
        int sample(){
            std::default_random_engine gen;
            std::poisson_distribution<int> dis(lambda);
            
            return dis(gen);
        }
        
    };
    
    class Exponential
    {
    protected:
        double lambda;
        
    public:
        
        const int n_param = 1;
        const double e = 2.7182818284590452353;
        Exponential(double _lambda){
            lambda = _lambda;
        }
        //pdf at a point
        double pdf(double x){
            return (lambda*pow(e, -lambda*x));
        }
        
        double P_leq(double x){
            return (1-pow(e,-lambda*x));
        }
        
        double mean(){
            return 1/lambda;
        }
        double variance(){
            return( 1 / (lambda*lambda));
        }
        
        double sample(){
            default_random_engine gen;
            exponential_distribution dis(lambda);
            return dis(gen);
        }
    };
    
    class Normal
    {
    protected:
        double mu;
        double sigma;
    public:
        const double sroot2pi = 1.772453851;
        const double e = 2.7182818284590452353;
        const int n_param = 2;
        Normal(double _mu, double _sigma){
            mu = _mu;
            sigma = _sigma;
        }
        double P_leq(double x){
            return erf(-(x-mu)/(sqrt(2)*sigma))/2;
        }
        double pdf(double x){
            return pow(e, -((x-mu)*(x-mu))/(2*sigma*sigma))/(sroot2pi*sigma)
        }
        double mean(){
            return mu;
        }
        double variance(){
            return sigma*sigma;
        }
        
        double sample(){
            default_random_engine gen;
            normal_distribution dis(mu,sigma);
            return dis(gen);
        }
        
    };
    
}

#endif /* Distributions_hpp */
