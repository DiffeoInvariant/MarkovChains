#ifndef pIMH_hpp
#define pIMH_hpp

#include <stdio.h>
#include<random>
#include<numeric>
#include<vector>
#include<array>
#include<utility>
#include "Distributions.h"

using namespace std;
namespace Markov
{

    /**
     *@author: Zane Jakobs
     *@brief: a class to run the perfect independent Metropolis Hastings algorithm
     developed by Jem Corcoran and R.L. Tweedie. Original paper at
     https://projecteuclid.org/euclid.aoap/1015345299#abstract
     */
    template<typename CandidateDist, typename TargetDist>
    class IMH
    {
    protected:
        /*
         1.11 is default so we can check if it hasn't been initialized
         
         Usually, best practices would be to use boost::optional<double>
        if you don't have C++17, or some of your program is old and
         won't work with new std types, or std::optional<double>
         if you're compiling with C++17 or later. For this class though, we're going
         to use a preset value to reduce overhead, since speed is everything
         in perfect sampling land.*/
        double lower_bound{ 1.11 };
        CandidateDist Q;
        TargetDist pi;

    public:
        
        constexpr IMH(const CandidateDist& _Q, const TargetDist& _pi,double lb) : Q(_Q), pi(_pi), lower_bound(lb) {}
        
        
        constexpr auto MH_ratio(const double x, const double y) const noexcept
        {
            //std::cout << pi.pdf(y) << " " << Q.pdf(x) << "\n";
            return (pi.pdf(y)*Q.pdf(x))/(pi.pdf(x)*Q.pdf(y));
        }
        
        /**
         *@author: Zane Jakobs
         *@algorithm by Corcoran and Tweedie
         *@return: "larger" of x and y according to the partial order for perfect IMH
         */
        constexpr auto partial_order(const double x, const double y) const noexcept
        {
            return MH_ratio( x, y) >= 1 ? x : y;
        }
        
        /**
         *@author: Zane Jakobs
         * @brief: alpha(x,y) in the paper
         *@return: min(1, MH_ratio)
         */
        constexpr auto accceptance_threshold(const double x, const double y) const noexcept
        {
            auto ratio = MH_ratio(x, y);
            return ratio < 1.0 ? ratio : static_cast<double>(1.0);
        }
        /**
         *@author: Zane Jakobs
         *@brief: finds \ell from the paper, lower bound on reordered sample space
         */
        //constexpr auto find_lower_bound(const TargetDist& _pi) const noexcept;
        
        
        /**
         *@author: Zane Jakobs
         *@brief: runs the classical Metropolis Hastings algorithm with
         * a symmetric transition kernel from time t = -n to 0 with pre-chosen
         * samples from the uniform and from the candidate
         */
        constexpr auto MH_from_past(int& n, const std::vector<double>& qvec, const std::vector<double>& avec) const noexcept
        {
            
            auto vlen = qvec.size() - 1;
            
            //static_assert(vlen == avec.size());
            
            auto state = lower_bound;
            for(int t = 0; t <= n; t++){
                //compiler will optimize this to not declare a new one each loop
                auto threshold = accceptance_threshold(state, qvec[vlen - n +t] );
                
                if(avec[vlen - n + t] < threshold){
                    state = qvec[vlen - n + t];
                }
            }//end for
            return state;
        }
        
        /**
         *@author: Zane Jakobs
         *@brief: runs the perfect IMH algorithm once
         */
        auto perfect_IMH_sample(unsigned initial_len = 100) const noexcept
        {
            
            auto avec = Markov::uniform_sample_vector(initial_len, 0.0, 1.0);
            auto qvec = Q.create_sample_vector(initial_len);
            bool accepted_first = false;
            
            int n = 1;
            /*
             parallelize this loop!
             */
            while(!accepted_first){
                //vlen is "time 0"
                auto vlen = avec.size() - 1;
                //update vectors if we hit the end of them
                if(n == vlen){
                    avec = Markov::update_uniform_sample_vector(avec, initial_len, 0.0, 1.0);
                    Q.update_sample_vector(qvec, initial_len);
                    
                    vlen += initial_len;
                }
               // std::cout << "lb" << lower_bound << "\n";
                //std::cout << "q" << qvec[vlen - n] << "\n";

                auto threshold = accceptance_threshold(-lower_bound, qvec[vlen - n]);
                //std::cout << threshold << "\n";
                //if the first transition from time -n is accepted, we have converged
                if(avec[vlen - n] < threshold){
                    accepted_first = true;
                } else{
                    n++;//if we reject the transition, move back one in time
                }
            }//end while
            auto sample = MH_from_past(n, qvec, avec);
            return sample;
        }
    };
    
}//end namespace scope
#endif /* MetropolisHastings_hpp */
