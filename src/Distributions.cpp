#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cstdlib>
#include "../include/Distributions.h"  
using namespace std;
namespace Markov {
    
    /**
     *@author: Emily Jakobs
     *@return: reference to std::pair containing a uniform(0,1) distribution of type double and a random engine
     */
    pair<default_random_engine, uniform_real_distribution<double> > std_sampler_pair() noexcept
    {
        random_device rd;
        default_random_engine gen(rd());
        uniform_real_distribution<double> dis(0,1);
        return std::make_pair(gen, dis);
    }
    
    /**
     *@brief returns array of n samples from U(0,1) distribution
     *@author: Emily Jakobs
     *@param T: type of sample (double, float, etc)
     *@param n: length of sample
     *@return: array of n samples from uniform(0,1) distribution
     */
    double uniform_sample(pair<default_random_engine, uniform_real_distribution<double> >& spair) noexcept
    {
       // static_assert(is_arithmetic<T>::value, "Error: uniform_sample must take an arithmetic type");
        return spair.second(spair.first); //  return dis(gen)
    }
    vector<double> uniform_sample_vector(pair<default_random_engine, uniform_real_distribution<double> >& spar, size_t length) noexcept
    {
        vector<double> vec(length, 0.0);
        for(auto &i : vec){
            i = spar.second(spar.first);
        }
        return vec;
    }
    
    vector<double> update_uniform_sample_vector(vector<double>& sample_seq, pair<default_random_engine, uniform_real_distribution<double> >& spar, unsigned additions) noexcept
    {
        for(unsigned i = 0; i < additions; i++){
            auto it = sample_seq.begin();
            sample_seq.insert(it, spar.second(spar.first));
        }
        return sample_seq;
    }

    constexpr double Poisson::getLambda() noexcept{
            return lambda;
    }
    constexpr void Poisson::setLambda(double _lambda) noexcept{
        lambda = _lambda;
    }
        
    constexpr double Poisson::mean() noexcept{
        return lambda;
    }
    
    constexpr double Poisson::variance() noexcept{
        return lambda;
    }
    //let X ~ \Lambda(k). This function returns P(X = x)
    constexpr double Poisson::P_eq(const int& x) noexcept{
       // double p = pow(M_E, -lambda)*pow(lambda,double(x))/TMP_factorial<x>;
        return static_cast<double>(-1);//p;
    }
    //returns P( X <= x)
    constexpr double Poisson::cdf(const int& x) noexcept{
        double sum = 0;
        for(int i = 0; i <= x; i++){
            sum += P_eq(i);
        }
        return sum;
    }
    //generates a sample
    int Poisson::sample() noexcept{
        random_device rd;
        std::default_random_engine gen(rd());
        std::poisson_distribution<int> dis(lambda);
        
        return dis(gen);
    }
   
    
    constexpr void Exponential::setLambda(double _lambda) noexcept{
        lambda = _lambda;
    }
    double Exponential::pdf(double x) noexcept{
        return (lambda*pow(M_E, -lambda*x));
    }
        
     double Exponential::cdf(double x) noexcept{
        return (1-pow(M_E,-lambda*x));
    }
        
    constexpr double Exponential::mean() noexcept{
        return 1/lambda;
    }
    
    constexpr double Exponential::variance() noexcept{
        return( 1 / (lambda*lambda));
    }
        
    double Exponential::sample() noexcept {
        random_device rd;
        default_random_engine gen(rd());
        exponential_distribution<double> dis(lambda);
        return dis(gen);
    }
    
    constexpr void Normal::setMu(double _mu) noexcept{
        mu = _mu;
    }
    
    constexpr void Normal::setSigma(double _sigma) noexcept{
        sigma = _sigma;
    }
    
    double Normal::pdf(double x) const noexcept{
        return pow(M_E, -((x-mu)*(x-mu))/(2*sigma*sigma))/(sroot2pi*sigma);
    }
    constexpr double Normal::mean() const noexcept{
        return mu;
    }
    constexpr double Normal::variance() const noexcept{
        return sigma*sigma;
    }
        
    double Normal::sample() const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dis(mu,sigma);
        return dis(gen);
    }
    
    /**
     *@author: Emily Jakobs
     *@return: vector of samples from a normal distribution
     */
    vector<double> Normal::create_sample_vector(unsigned length) const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dis(mu,sigma);
        vector<double> vec(length);
        for(auto &i : vec){
            i = dis(gen);
        }
        return vec;
    }
    
    /**
     *@author: Emily Jakobs
     *@param additions: how many samples to push_back?
     *@brief: vector with additions many new samples from a Normal(mu,sigma) distribution.
     */
    void Normal::update_sample_vector(vector<double>& sample_seq, unsigned additions) const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dis(mu,sigma);
        
        for(unsigned i = 0; i < additions; i++){
            sample_seq.push_back(dis(gen));
        }
    }
    
    
    constexpr void Gamma::setAlpha(double a) noexcept{
        alpha = a;
    }
    
    constexpr void Gamma::setBeta(double b) noexcept{
        beta = b;
    }
    
    constexpr double Gamma::getAlpha() noexcept{
        return alpha;
    }
    
    constexpr double Gamma::getBeta() noexcept{
        return beta;
    }
    
    const double Gamma::pdf(double x) const noexcept{
        return (pow(x,alpha - 1)*pow(M_E, -x/beta)/(tgamma(alpha)*pow(beta,alpha)));
    }
    constexpr double Gamma::mean() noexcept{
        return alpha*beta;
    }
    constexpr double Gamma::variance() noexcept{
        return (alpha*beta*beta);
    }
    double Gamma::sample() noexcept{
        random_device rd;
        default_random_engine gen(rd());
        gamma_distribution<double> dis(alpha,beta);
        return dis(gen);
    }
    
    constexpr void AsymmetricStudentT::setLocation(double _loc) noexcept{
        location = _loc;
    }
    
    constexpr void AsymmetricStudentT::setScale(double _sc) noexcept{
        scale = _sc;
    }
    
    constexpr void AsymmetricStudentT::setSkew(double _sk) noexcept{
        skew = _sk;
    }
    
    constexpr void AsymmetricStudentT::setLTail(double _lt) noexcept{
        ltail = _lt;
    }
    
    constexpr void AsymmetricStudentT::setRTail(double _rt) noexcept{
        rtail = _rt;
    }
    /*
    constexpr double AsymmetricStudentT::getLocation() const noexcept{
        return location;
    }
    
    constexpr double AsymmetricStudentT::getScale() const noexcept{
        return scale;
    }
    
    constexpr double AsymmetricStudentT::getSkew() const noexcept{
        return skew;
    }
    
    constexpr double AsymmetricStudentT::getLTail() const noexcept{
        return ltail;
    }
    
    constexpr double AsymmetricStudentT::getRTail() const noexcept{
        return rtail;
    }
    */
    double AsymmetricStudentT::K(double v) const noexcept{
        auto kv = tgamma((v+1.0)*0.5) / (sqrt(M_PI * v)*tgamma(v*0.5));
        return kv;
    }
    
    double AsymmetricStudentT::getAStar() const noexcept{
        auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
        return astar;
    }
    
    //standard pdf, where location = 0, scale = 1
    double AsymmetricStudentT::std_pdf(double x) const noexcept{
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
    double AsymmetricStudentT::slow_pdf(double x) const noexcept{
        auto stdX = static_cast<double>((x-location)/scale);
        return (std_pdf(stdX)/scale);
    }
    
    //standard pdf, where location = 0, scale = 1
    double AsymmetricStudentT::fast_std_pdf(double x) const noexcept{
        auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
        auto normalize_left = astar*(1-skew)/( skew* (1-astar));
        if( x <= 0){
            auto fx = pow((1+ (1.0/ltail) * (x*x)/(4*astar*astar)), -0.5*(1+ltail));
            return normalize_left*fx;
        }
        else{
            auto fx = pow( (1.0 + (x*x)/(4 * rtail * (1.0-astar)*(1.0-astar))), -0.5*(1+rtail));
            return normalize_left*fx;
        }
    }
    
    double AsymmetricStudentT::pdf(double x)const  noexcept{
        auto stdX = static_cast<double>((x-location)/scale);
        return (fast_std_pdf(stdX));
    }
    double AsymmetricStudentT::cdf(double x) const noexcept{
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
    
    
    double AsymmetricStudentT::inv_cdf(double x) const noexcept{
        if(x == skew){
            return static_cast<double>(0);
        }
        
        double inv_stdcdf_l, inv_stdcdf_2;
        auto astar = skew*K(ltail)/ ( skew*K(ltail) + (1-skew) * K(rtail));
        if( x > skew){
            inv_stdcdf_l = 0.0;
        } else{
            inv_stdcdf_l = gsl_cdf_tdist_Pinv( x/(2.0 *skew), ltail);
        }//end else
        
        if( x > 1.0 - skew){
            inv_stdcdf_2 = gsl_cdf_tdist_Pinv( x/(2.0 *(1.0 - skew)), rtail);
        } else{
            inv_stdcdf_2 = 0.0;
        }//end else
        return static_cast<double>(2.0 * astar * inv_stdcdf_l + 2.0 * (1.0-astar)* inv_stdcdf_2);
    }
    
    constexpr void Cauchy::setMu(double _mu) noexcept{mu = _mu;}
    
    constexpr void Cauchy::setSigma(double s) noexcept{sigma = s;}
    
    //constexpr double Cauchy::pdf(double x)const noexcept{
    //    double adjX = (x-mu)/sigma;
   //     return 1.0/(M_PI*sigma* (1+adjX*adjX));
   // }
    
    double Cauchy::sample() const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        cauchy_distribution<double> dis(mu,sigma);
        return abv ? abs(dis(gen)) : dis(gen);
    }
    
    /**
     *@author: Emily Jakobs
     *@return: vector of samples from a normal distribution
     */
    vector<double> Cauchy::create_sample_vector(unsigned length) const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        cauchy_distribution<double> dis(mu,sigma);
        vector<double> vec(length);
        for(auto &i : vec){
            i = abv ? abs(dis(gen)) : dis(gen);
        }
        return vec;
    }
    
    /**
     *@author: Emily Jakobs
     *@param additions: how many samples to push_back?
     *@brief: vector with additions many new samples from a Normal(mu,sigma) distribution.
     */
    void Cauchy::update_sample_vector(vector<double>& sample_seq, unsigned additions) const noexcept{
        random_device rd;
        default_random_engine gen(rd());
        cauchy_distribution<double> dis(mu,sigma);
        
        for(unsigned i = 0; i < additions; i++){
            auto spl = abv ? abs(dis(gen)) : dis(gen);
            sample_seq.push_back(spl);
        }
    }
    
}

