//
//  MarkovHW11.cpp
//  
//
//  Created by Zane Jakobs on 4/25/19.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Markov/pIMH.h>
#include <Markov/Distributions.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
class Target
{
    
public:
    constexpr Target() {};
    const auto pdf(double x) const noexcept
    {
        constexpr double twoNinths = 0.2222222222;
        return x > 0 ?
        twoNinths * x * std::pow(M_E, -1.0 * x * x * 0.1111111111) :
        0.0;
    }
};

int main()
{
    const auto targ = Target();
    
    const auto candidate = Markov::Cauchy(2.5,7.0,true);
    constexpr double lowestState = 2.08355;
    
    auto sampler = Markov::IMH(candidate,targ,lowestState);
    
    constexpr std::size_t numSamples = 1.0e6;
    
    auto samples = sampler.perfect_IMH_sample_vector(numSamples);
    
    //write to file for plotting with Python and seaborn
    std::ofstream fileWriter;
    fileWriter.open("HW11_samples.csv");
    fileWriter << "Data\n";
    for(auto &i : samples){
        fileWriter << i << "\n";
    }
    fileWriter.close();
    
    return 0;
}


