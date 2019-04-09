
#ifndef Distributions_hpp
#define Distributions_hpp
#include<utility>
#include<array>
#include<vector>
#include<random>
#include<type_traits>
/**
 NOTE FROM AUTHOR (ZANE JAKOBS):
 ONLY KEEP #include<optional> IF YOU ARE COMPILING WITH C++17 OR LATER
IF YOU ARE COMPILING WITH C++14 OR EARLIER, USE std::pair<T,bool> TO
 RECREATE THE FUNCTIONALITY OF std::optional<T>; THAT IS, THE SIGNATURE
 template<typename T>
 auto func(std::optional<T> param) { param ? return param
 */
#include<optional>
using namespace std;
//using namespace Markov;
namespace Markov {
    
    /**
     *@author: Zane Jakobs
     *@param T: floating-point type of sample
     *@return: reference to std::pair containing a uniform(0,1) distribution of type T and a random engine
     */
    pair<default_random_engine, uniform_real_distribution<double> > std_sampler_pair() noexcept;
    

    double uniform_sample(pair<default_random_engine, uniform_real_distribution<double> >& spair) noexcept;
    /**
     *@brief returns array of n samples from U(0,1) distribution
     *@author: Zane Jakobs
     *@param T: type of sample (double, float, etc)
     *@param n: length of sample
     *@return: array of n samples from uniform(0,1) distribution
     */
   // template<typename T, size_t N>
   // array<T,N> uniform_sample_arr() noexcept;
    template<typename T, size_t N>
    array<T,N> uniform_sample_arr() noexcept
    {
        static_assert(is_arithmetic<T>::value, "Error: uniform_sample_arr must take an arithmetic type");
        auto ed_pair = std_sampler_pair();
        array<T,N> arr;
        for(auto &i : arr){
            i = uniform_sample(ed_pair);
        }
        return arr;
    }
    
    /**
     *@author: Zane Jakobs
     *@return: length-many samples from a U(lower, upper) distribution.
     */
   // template<typename T>
   // vector<T> uniform_sample_vector(size_t length, T lower, T upper) noexcept;
    template<typename T>
    vector<T> uniform_sample_vector(size_t length, T lower, T upper) noexcept
    {
        static_assert(is_arithmetic<T>::value, "Error: uniform_sample_vector must take an arithmetic type");
        random_device rd;
        default_random_engine gen(rd());
        uniform_real_distribution<T> dis(lower,upper);
        vector<T> vec(length, 0.0);
        for(auto &i : vec){
            i = dis(gen);
        }
        return vec;
    }
    
    /**
     *@author: Zane Jakobs
     *@brief: vector with additions-many more samples from a U(lower, upper) distribution.
     */
   // template<typename T>
    //void update_uniform_sample_vector(vector<T>& sample_seq, unsigned additions, T lower, T upper) noexcept;
    template<typename T>
    auto update_uniform_sample_vector(vector<T>& sample_seq, unsigned additions, T lower, T upper) noexcept{
        random_device rd;
        default_random_engine gen(rd());
        uniform_real_distribution<T> dis(lower,upper);
        for(unsigned i = 0; i < additions; i++){
            auto it = sample_seq.begin();
            sample_seq.insert(it, dis(gen));
        }
        return sample_seq;
    }
    /**
     *@author: Zane Jakobs
     *@param inv_cdf: inverse CDF function
     *@preconditon: iCDfunc must return double, or a type that can be implicitly converted
     *@return: returns array of n independent samples from distribution by taking F^{-1}(x), x ~ Unif(0,1)
     */
   // template<typename inv_cdf, size_t N>
    //constexpr array<double> sample_arr_ind(inv_cdf&& iCDfunc, unsigned n) noexcept;
        
    template<typename inv_cdf, std::size_t N>
    constexpr array<double, N> sample_arr_ind(inv_cdf&& iCDfunc) noexcept
    {
        auto unif_arr = uniform_sample_arr<double,N>();
        for(auto &i : unif_arr){
            i = iCDfunc(i);
        }
        return unif_arr;
    }
    
    class Poisson
    {
    protected:
        double lambda = 1.0; //lambda  = rate
        
    public:
        const int n_param = 1; //1 parameter for this class

        Poisson() {}
        
        constexpr Poisson(double _lambda){
            lambda = _lambda;
        }
        
        constexpr double getLambda() noexcept;

        constexpr void setLambda(double _lambda) noexcept;
        
        constexpr double mean() noexcept;
        
        constexpr double variance() noexcept;
        
        //let X ~ \Lambda(k). This function returns P(X = x)
        constexpr double P_eq(const int& x) noexcept;
        //returns P( X <= x)
        constexpr double cdf(const int& x) noexcept;
        //generates a sample
        int sample() noexcept;
        
    };
    
    class Exponential
    {
    protected:
        double lambda = 1.0;
        
    public:
        const int n_param = 1;
        Exponential() {lambda = 1.0;}
        
        constexpr Exponential(double l){
            lambda = l;
        }
        
        constexpr void setLambda(double _lambda) noexcept;

        //pdf at a point
        double pdf(double x) noexcept;
        
        double cdf(double x) noexcept;
        
        constexpr double mean() noexcept;
        
        constexpr double variance() noexcept;
        
        double sample() noexcept;
    };
    
    class Normal
    {
    protected:
        double mu = 0.0;
        double sigma = 1.0;
    public:
        const int n_param = 2;
        
        constexpr Normal(double m = 0.0, double s = 1.0){
            mu = m;
            sigma = s;
        }
        //sqrt(2*pi)
        const double sroot2pi = 1.772453851;
    
        constexpr void setMu(double _mu) noexcept;
        
        constexpr void setSigma(double _sigma) noexcept;
        
        double pdf(double x) const noexcept;
        
        constexpr double mean() const noexcept;
        
        constexpr double variance() const noexcept;
        
        double sample() const noexcept;
        
        /**
         *@author: Zane Jakobs
         *@return: vector of samples from a normal distribution
         */
        vector<double> create_sample_vector(unsigned length) const noexcept;
        
        /**
         *@author: Zane Jakobs
         *@param additions: how many samples to push_back?
         *@brief: vector with additions many new samples from a Normal(mu,sigma) distribution.
         */
        void update_sample_vector(vector<double>& sample_seq, unsigned additions) const noexcept;
    };
    
    class Gamma
    {
    protected:
        //alpha and beta are distribution paameters
        double alpha = 1.0;
        double beta = 1.0;
    public:
        
        constexpr Gamma(double a = 1.0, double b = 1.0){
            alpha = a;
            beta = b;
        }
        constexpr void setAlpha(double a) noexcept;
        
        constexpr void setBeta(double b) noexcept;
        
        constexpr double getAlpha() noexcept;
        
        constexpr double getBeta() noexcept;
        
        double pdf(double x) noexcept;
        
        constexpr double mean() noexcept;
        
        constexpr double variance() noexcept;
        
        double sample() noexcept;

    };
    
    
    
    /**
     *@author: Zane Jakobs
     *see https://pdfs.semanticscholar.org/d91f/bda26e5824717245e35621e961885cbee2b3.pdf
     * particularly useful for econometric modeling with GAS models
     */
    class AsymmetricStudentT
    {
    protected:
        double location;
        double scale;
        double skew;
        double ltail;
        double rtail;
    public:
        const int n_param = 5;
        
        constexpr AsymmetricStudentT(double lc, double sc,double sk, double lt, double rt) : location(lc), scale(sc), skew(sk), ltail(lt), rtail(rt) {}
        
        
        constexpr void setLocation(double _loc) noexcept;
        
        constexpr void setScale(double _sc) noexcept;
        
        constexpr void setSkew(double _sk) noexcept;
        
        constexpr void setLTail(double _lt) noexcept;
        
        constexpr void setRTail(double _rt) noexcept;
        
        constexpr double getLocation() const noexcept{return location;}
        
        constexpr double getScale() const noexcept {return scale;}
        
        constexpr double getSkew() const noexcept {return skew;}
        
        constexpr double getLTail() const noexcept {return ltail;}
        
        constexpr double getRTail() const noexcept {return rtail;}
        
        
        //helper function, as defined in the paper in comment at class declaration
        double K(double v) const noexcept;
        
        double getAStar() const noexcept;
        //standard pdf, where location = 0, scale = 1
        double std_pdf(double x) const noexcept;
        
        //general form: pdf(x) = (1/scale)*std_pdf( (x-location)/scale).
        //THE ACTUAL PDF
        double slow_pdf(double x) const noexcept;
        //a quicker-to-evaluate, but it's the actual pdf
        double fast_std_pdf(double x) const noexcept;
        //NOT THE ACTUAL PDF. it is proportional to it though.
        double pdf(double x) const noexcept;
        
        double cdf(double x) const noexcept;
        
        
        double inv_cdf(double x) const noexcept;
        
        //minimum point wrt the IMH order when using gaussian 0,1 innovations
        //constexpr auto Gaussian_IMH_min_point() noexcept;

    };
    
    
    class Cauchy
    {
    protected:
        double mu = 0.0;
        double sigma = 1.0;
    public:
        const int n_param = 2;
        
        constexpr Cauchy(double m = 0.0, double s = 1.0){
            mu = m;
            sigma = s;
        }
        
        constexpr void setMu(double _mu) noexcept;
        
        constexpr void setSigma(double _sigma) noexcept;
        
        constexpr double pdf(double x) const noexcept
        {
            double adjX = (x-mu)/sigma;
            return 1.0/(M_PI*sigma* (1+adjX*adjX));
        }
        
        double sample() const noexcept;
        
        /**
         *@author: Zane Jakobs
         *@return: vector of samples from a Cauchy distribution
         */
        vector<double> create_sample_vector(unsigned length) const noexcept;
        
        /**
         *@author: Zane Jakobs
         *@param additions: how many samples to push_back?
         *@brief: vector with additions many new samples from a Cauchy(mu,sigma) distribution.
         */
        void update_sample_vector(vector<double>& sample_seq, unsigned additions) const noexcept;
    };
    
    
}

#endif /* Distributions_hpp */
