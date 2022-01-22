//
//  distTest.cpp
//  
//
//  Created by Emily Jakobs on 4/6/19.
//
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include<mkl.h>//optimizations
#include <stdio.h>
#include <stdlib.h>
#include<random>
#include<numeric>
#include<type_traits>
#include<gsl/gsl_cdf.h>
#include<utility>
#include<array>
#include<vector>
#include "Distributions.h"
#include<iostream>
#include<sstream>
#include<fstream>
#include<typeinfo>
#include<string>
#include<ctime>
#include<gsl/gsl_histogram.h>
#include<algorithm>
#include"pIMH.h"

template<typename S, typename T, typename = void>
struct is_to_stream_writable: std::false_type {};

template<typename S, typename T>
struct is_to_stream_writable<S, T,
std::void_t<  decltype( std::declval<S&>()<<std::declval<T>() )  > >
: std::true_type {};

/*
template<typename testFunc, class ... ArgTypes>
auto funcTest(testFunc&& f, std::string name, ArgTypes ... args){
    std::cout << "Testing " << name << "/n";
    auto ret = f(args...);
    if( is_to_stream_writable<std::ostream, decltype(ret)>::value ){
        std::cout << name << " returned " << ret;
        return std::make_pair(ret, true);
    } else{
        std::cout << name << " has a return type that cannot be streamed, but it did not crash. /n";
        return std::make_pair(ret, false);
    }
}
*/

int main(){
    auto sampler = Markov::std_sampler_pair();
    
    auto uniSample = Markov::uniform_sample(sampler);
    std::cout << "Testing uniform_sample: \n";
    std::cout << uniSample << endl;
    std::cout << "Testing Cauchy() and Cauchy::Sample() \n";
    const auto candidate = Markov::Cauchy(0.0,2.0);
    std::cout << candidate.sample();
    std::cout << "\n Testing uniform_sample_vector<double>: \n";
    const std::size_t arrSize = 10;
    auto uniVec = Markov::uniform_sample_vector(arrSize, (double)0.0, (double)1.0);
    for(auto &i : uniVec){
       std::cout << i << endl;
    }
    
    std::cout << "Testing uniform_sample_arr<double>: \n";
    //std::size_t arrSize = 10;
    auto utime1 = std::chrono::high_resolution_clock::now();
    auto uniArr = Markov::uniform_sample_arr<double, arrSize>();
    for(auto &j : uniArr){
        std::cout << j << endl;
    }
    auto utime2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time in microseconds: \n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(utime2 - utime1).count() << "\n";
    std::cout << "Creating an AST, printing f_{AST}(0) \n";
    /**
      Keep the third parameter (skewness paramater) between 0 and 1, and
     keep the last two parameters (tail exponent parameters) between 4.0 and 50.0. Mean and standard deviation (first and second parameters) can be anything reasonable.
     */
    const auto ast = Markov::AsymmetricStudentT(0.00, 4.00, 0.58, 5.0, 7.8);
    std::cout << ast.slow_pdf(0.0) << "\n";
    
    std::cout << "Passing uniform samples through inverse AST cdf: \n";
    auto atime1 = std::chrono::high_resolution_clock::now();
    for(auto &i : uniArr){
        std::cout << ast.inv_cdf(i) << "\n";
    }
    auto atime2 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Time in microseconds: \n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(atime2 - atime1).count() << "\n";
    
    std::cout << "Testing IMH constructor: \n";
    double lb = 0.0;
    auto perfectIMH = Markov::IMH(candidate,ast,lb);
    std::cout << "Testing IMH MH_ratio(1.0, 100.0): \n";
    std::cout << perfectIMH.MH_ratio(1.0,1.4);
    std::cout << "Testing IMH partial_order(1.0,100.0): \n";
    std::cout << perfectIMH.partial_order(1.0,1.4);
    std::cout << "Testing IMH perfect_IMH_sample(): \n";
    std::array<double,20000> svec;
    for(int i = 0; i < 20000; i++){
        svec[i] = perfectIMH.perfect_IMH_sample(1000);
    }
    
    std::ofstream fileWriter;
    fileWriter.open("IMH_samples.csv");
    fileWriter << "Data\n";
    for(auto &i : svec){
        fileWriter << i << "\n";
    }
    fileWriter.close();
    
    
    auto maxId = std::distance(svec.begin(), std::max_element(svec.begin(), svec.end()) );
    auto minId =std::distance(svec.begin(), std::min_element(svec.begin(), svec.end()) );
    double max = svec[maxId];
    double min = svec[minId];
    size_t nbins = 4;
    auto hist = gsl_histogram_alloc(nbins);
    auto hSet = gsl_histogram_set_ranges_uniform(hist, min, max);
    for(auto &i: svec){
        //std::cout << i << "\n";
        gsl_histogram_increment(hist, i);
    }
    //gsl_histogram_fprintf(stdout, hist, "%g", "%g");
    gsl_histogram_free(hist);
    return 0;
    
}
