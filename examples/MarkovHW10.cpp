//
//  MarkovHW10.cpp
//  
//
//  Created by Zane Jakobs on 4/23/19.
//

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include<random>
#include<numeric>
#include<type_traits>
#include<utility>
#include<array>
#include<vector>
#include <Markov/Distributions.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<typeinfo>
#include<string>
#include<ctime>
#include<algorithm>
#include <Markov/pIMH.h>

int main(){
    //absolute value (setting last param = true) Cauchy distribution
    //with location 1.5, scale 1.5
    const auto candidate = Markov::Cauchy(2.5, 1.0, true);
    //Gamma(3,2)
    const auto target = Markov::Gamma(3.0,0.5);
    /*
     the Mathematica code to find the max of Gamma(3,2)/Cauchy(1.5,1.0) is
     f[x_] = PDF[GammaDistribution[3, 2], x];
     g[x_] = Abs[PDF[CauchyDistribution[1.5, 1.0], x]];
     h[x_] = f[x]/g[x];
     NMaximize[h[x], x, Reals]
     
     which returns x -> 0.701872

     */
    double lowestState = 0.701872;
    auto IMHSampler = Markov::IMH(candidate, target, lowestState);
    constexpr std::size_t num_samples = 1.0e6; //take 10,000 samples
    #pragma omp parallel num_threads(4)
    auto samples = IMHSampler.perfect_IMH_sample_vector(num_samples);
    
    //write to file for plotting with Python and seaborn
    std::ofstream fileWriter;
    fileWriter.open("HW10_samples.csv");
    fileWriter << "Data\n";
    for(auto &i : samples){
        fileWriter << i << "\n";
    }
    fileWriter.close();
    
    return 0;
}
