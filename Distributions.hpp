
#ifndef Distributions_hpp
#define Distributions_hpp
#define _USE_MATH_DEFINES
#include<mkl.h>//optimizations
#include<stdio.h>
#include<random>
#include<cmath>
#include<limits>
#include<gsl_cdf.h>
#include<utility>
#include<array>
#include<vector>
using namespace std;
using namespace Markov;
namespace Markov {
    
    /**
     *@author: Zane Jakobs
     *@param T: floating-point type of sample
     *@return: reference to std::pair containing a uniform(0,1) distribution of type T and a random engine
     */
    template<typename T>
    constexpr auto std_sampler_pair()
    {
        default_random_engine gen;
        uniform_real_distribution<T> dis(0,1);
        pair<default_random_engine, uniform_real_distribution<T> > par(gen, dis);
        return par;
    }
    
    /**
     *@brief returns array of n samples from U(0,1) distribution
     *@author: Zane Jakobs
     *@param T: type of sample (double, float, etc)
     *@param n: length of sample
     *@return: array of n samples from uniform(0,1) distribution
     */
    template<typename T>
    constexpr T uniform_sample(pair<default_random_engine, uniform_real_distribution<T> >& spair)
    {
        return spair.second(spair.first); //  return dis(gen)
    }
    //array of length n containing samples
    template<typename T>
    constexpr auto uniform_sample_arr(const unsigned n)
    {
        auto ed_pair = std_sampler_pair<T>();
        array<T,n> arr;
        for(auto &i : arr){
            i = uniform_sample<T>(ed_pair);
        }
        return arr;
    }
    //returns array of n independent samples from distribution
    //by taking F^{-1}(x), x ~ Unif(0,1)
    /**
     *@author: Zane Jakobs
     *@param inv_cdf: inverse CDF function
     *@preconditon: iCDfunc must return double, or a type that can be implicitly converted
     * to double
     */
    template<typename inv_cdf>
    constexpr auto sample_arr_ind(const unsigned n, inv_cdf&& iCDfunc)
    {
        auto unif_arr = uniform_sample_arr<double>(n);
        for(auto &i : unif_arr){
            i = iCDfunc(i);
        }
        return unif_arr;
    }
    
    
    
    template<unsigned n>
    struct TMP_factorial
    {
        enum {value = n*TMP_factorial<n-1>::value };
    };
    template<>
    struct TMP_factorial<0>
    {
        enum { value = 1 };
    };
    class Poisson
    {
    protected:
        double lambda = 1.0; //lambda  = rate
        
    public:
        const int n_param = 1; //1 parameter for this class
        //default to largest possible lambda if no argument is passed
        Poisson(double _lambda){
            lambda = _lambda;
        }
        
        double getLambda(){
            return lambda;
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
        double P_eq(const int& x){
            //double p = pow(M_E, -lambda)*pow(lambda,double(x))/TMP_factorial<x>;
            return -1;
        }
        //returns P( X <= x)
        double P_leq(const int& x){
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
        double lambda = 1.0;
        
    public:
        constexpr Exponential(double l = 1.0){
            lambda = l;
        }
        
        constexpr int n_param = 1;
        //pdf at a point
        constexpr double pdf(double x){
            return (lambda*pow(M_E, -lambda*x));
        }
        
        constexpr double P_leq(double x){
            return (1-pow(M_E,-lambda*x));
        }
        
        constexpr double mean(){
            return 1/lambda;
        }
        constexpr double variance(){
            return( 1 / (lambda*lambda));
        }
        
        constexpr double sample(){
            default_random_engine gen;
            exponential_distribution<double> dis(lambda);
            return dis(gen);
        }
    };
    
    class Normal
    {
    protected:
        double mu = 0.0;
        double sigma = 1.0;
    public:
        constexpr Normal(double m = 0.0, double s = 1.0){
            mu = m;
            sigma = s;
        }
        
        constexpr double sroot2pi = 1.772453851;
        constexpr int n_param = 2;
    
        constexpr double pdf(double x){
            return pow(M_E, -((x-mu)*(x-mu))/(2*sigma*sigma))/(sroot2pi*sigma);
        }
        constexpr double mean(){
            return mu;
        }
        constexpr double variance(){
            return sigma*sigma;
        }
        
        constexpr double sample(){
            default_random_engine gen;
            normal_distribution<double> dis(mu,sigma);
            return dis(gen);
        }
        
    };
    
    class Gamma
    {
    protected:
        //alpha and beta are distribution paameters
        double alpha;
        double beta;
    public:
        constexpr Gamma(double a = 1.0, double b = 1.0){
            alpha = a;
            beta = b;
        }
        
        
        constexpr double pdf(double x){
            return (pow(x,alpha - 1)*pow(M_E, -x/beta)/(tgamma(alpha)*pow(beta,alpha)));
        }
        constexpr double mean(){
            return alpha*beta;
        }
        constexpr double variance(){
            return (alpha*beta*beta);
        }
        constexpr double sample(){
            default_random_engine gen;
            gamma_distribution<double> dis(alpha,beta);
            return dis(gen);
        }
    };
    
    
    
    /**
     *@author: Zane Jakobs
     *see https://pdfs.semanticscholar.org/d91f/bda26e5824717245e35621e961885cbee2b3.pdf
     * particularly useful for econometric modeling with GAS models
     */
    class AsymmetricStudentT
    {
    protected:
        double location = 0.0, scale = 1.0, skew = 0.0, ltail, rtail;
    public:
        constexpr AsymmetricStudentT(double lc, double sc, sk, double lt, double rt){
            location = lc;
            scale = sc;
            skew = sk;
            ltail = lt;
            rtail = rt;
        }
        constexpr auto K(double& v){
            auto kv = tgamma((v+1.0)*0.5) / (sqrt(M_PI * v)*tgamma(v*0.5));
            return kv;
        }
        //standard pdf, where location = 0, scale = 1
        constexpr auto std_pdf(double& x){
            if( x <= 0){
                auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
                auto fx = (skew/astar)*K(ltail);
                fx *= pow((1+ (1.0/ltail) * (x*x)/(4*astar*astar)), -0.5*(1+ltail));
                return fx;
            }
            else{
                auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
                auto fx = ((1.0-skew)/(1.0-astar))*K(rtail);
                fx *= pow( (1.0 + (x*x)/(4 * rtail * (1.0-astar)*(1.0-astar))), -0.5*(1+rtail));
                return fx;
            }
        }
        
        //general form: pdf(x) = (1/scale)*std_pdf( (x-location)/scale)
        constexpr auto pdf(double& x){
            auto stdX = static_cast<double>((x-location)/scale);
            return (std_pdf(stdX)/scale);
        }
        
        constexpr auto cdf(double& x){
            if(x == 0){
                return skew;
            }
            auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1.0-skew) * K(rtail));
            if( x > 0){
                //student-t CDF:
                auto stcdf = gsl_cdf_tdist_P( x/ (2.0*(1.0-astar)), rtail);
                auto F_AST = 2*(1-skew)*(stcdf - 0.5) + skew;
                return static_cast<double>(F_AST);
            }
            else{
                auto stcdf = gsl_cdf_tdist_P( x/ (2.0*astar), ltail);
                return static_cast<double>(2.0 *skew*stcdf);
            }
        }
        
        
        constexpr auto inv_cdf(double& x){
            if(x == skew){
                return static_cast<double>(0);
            }
            auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
            if( x > skew){
                auto inv_stdcdf_l = 0.0;
            } else{
                auto inv_stdcdf_l = gsl_cdf_tdist_Pinv( x/(2.0 *skew), ltail);
            }//end else
            
            if( x > 1.0 - skew){
                auto inv_stdcdf_2 = gsl_cdf_tdist_Pinv( x/(2.0 *(1.0 - skew)), rtail);
            } else{
                auto inv_stdcdf_2 = 0.0;
            }//end else
            return static_cast<double>(2.0 * astar * inv_stdcdf_l + 2.0 * (1.0-astar)* inv_stdcdf_2);
        }
        
    };
    
}

#endif /* Distributions_hpp */
