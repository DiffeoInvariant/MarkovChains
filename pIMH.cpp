#include <stdio.h>
#include<random>
#include<numeric>
#include<vector>
#include<utility>
#include"Distributions.h"
#include "pIMH.h"

namespace Markov
{
    using sampler = std::pair<default_random_engine, uniform_real_distribution<double> >;
    /**
     GENERAL NB: pi is the target distribution, Q is the candidate
     */
    
    
    template<typename CandidateDist, typename TargetDist>
    constexpr auto IMH<CandidateDist, TargetDist>::MH_ratio(double x, double y) noexcept{
        return (pi.pdf(y)*Q.pdf(x))/(pi.pdf(x)*Q.pdf(y));
    }
    /**
    *@author: Zane Jakobs
     *@param candidate_pdf, target_pdf: functions to compute values of the target and candidate pdf
     * (or at least something proportional to the target pdf)
     *@return: "larger" of x and y according to the partial order for perfect IMH. Returns x if x==y
     */
    template<typename CandidateDist, typename TargetDist>
    constexpr auto IMH<CandidateDist, TargetDist>::partial_order(double x, double y) noexcept{
        return MH_ratio( x, y) >= 1 ? x : y;
    }

    /**
     *@author: Zane Jakobs
     *@return: min(1, MH_ratio)
     */
    template<typename CandidateDist, typename TargetDist>
    constexpr auto IMH<CandidateDist, TargetDist>::accceptance_threshold(double x, double y) noexcept{
        auto ratio = MH_ratio(x, y);
        return ratio < 1.0 ? ratio : static_cast<double>(1.0);
    }
    
    constexpr auto IMH<AsymmetricStudentT, Normal>::find_lower_bound() noexcept{
        auto astar = pi.getAStar();
        auto scale = pi.getScale();
        //derivative of AST/N == 0 for AST having location = 0, scale = 1
        if(astar < 0.5){
            auto ltail =  pi.getLTail();
            auto k = pi.K(ltail)
            auto std_solution = sqrt(1 + ltail*( 1 - 4* astar * astar * scale * scale * k * k));
            return (std_solution + pi.location);
        } else{
            auto rtail = pi.getRTail();
            auto k = pi.K(rtail)
            auto std_solution = sqrt(1 + rtail*(1 - 4 * (1-astar) * (1-astar) * k * k * scale * scale));
            
            return (std_solution + pi.location);
        }
    }
    /**
     *@author: Zane Jakobs
     *@brief template specialization for normal candidate, AST target
     */
    constexpr auto IMH<Normal, AsymmetricStudentT>::IMH(const Normal& _Q,const AsymmetricStudentT& _pi){
        Q = _Q;
        pi = _pi;
        lower_bound = find_lower_bound();
    }
    
    
    template< typename TargetDist>
    constexpr auto IMH<Normal, TargetDist>::MH_from_past(int n, const std::vector<double>& qvec, const std::vector<double>& avec) noexcept{
        
        auto vlen = qvec.size() - 1;
        
        static_assert(vlen == avec.size());
        
        auto state = lower_bound;
        for(int t = 0; t <= n; t++){
            //compiler will optimize this to not declare a new one each loop
            auto threshold = accceptance_threshold(state, qvec[vlen - n +t] );
            
            if(avec[vlen - n + t] < threshold){
                state = q[vlen - n + t];
            }
        }//end for
        return state;
    }
    
    //initial len: length to preallocate sample arrays to
    template< typename TargetDist>
    constexpr auto IMH<Normal, TargetDist>::perfect_IMH_sample(unsigned initial_len, const sampler& spar) noexcept{
        
        auto avec = Markov::uniform_sample_vector<double>(initial_len, 0.0, 1.0);
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
                qvec = Q.update_sample_vector(qvec, initial_len, 0.0, 1.0);
                
                vlen += initial_len;
            }
            
            auto threshold = accceptance_threshold(lower_bound, qvec[vlen - n + 1]);
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
    
}//end namespace scope
