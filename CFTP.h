#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#ifndef CFTP_hpp
#define CFTP_hpp
#ifdef Success
#undef Success
#endif
#include <Eigen/Core>
#include<deque>
#include<valarray>
#include<random>
#include<algorithm>
#include<type_traits>
#include"MarkovChain.h"
#include<mkl.h>
#include<complex>
using namespace std;
using namespace Markov;
/**
 * Coupling From The Past to perfectly sample from a Markov chain matrix mat
 */
namespace Markov
{
    
    /**
     * @author: Zane Jakobs
     * @param mat: matrix to raise to power
     * @param _pow: power of matrix
     * @return: mat^_pow
     */
    Eigen::MatrixXd small_mat_pow(Eigen::MatrixXd mat, int _pow);
 
   
    //void *threadedMatPow(Eigen::MatrixXd &mat, int pow);
   
    double variation_distance(Eigen::MatrixXd dist1, Eigen::MatrixXd dist2);
    //pass by val for a cheap-ish copy
    double k_stationary_variation_distance(Eigen::MatrixXd trans, int k);
    int mixing_time(const Eigen::MatrixXd &trans);

    int isCoalesced(const Eigen::MatrixXd &mat);

    /**
     * @author: Zane Jakobs
     * @summary: voter CFTP algorithm to perfectly sample from the Markov chain with transition matrix mat. Algorithm from https://pdfs.semanticscholar.org/ef02/fd2d2b4d0a914eba5e4270be4161bcae8f81.pdf
     * @return: perfect sample from matrix's distribution
     */
    int voter_CFTP(const Eigen::MatrixXd &mat);

    /**
     *@author: Zane Jakobs
     * @param trans: transition matrix
     * @param gen, dis: random number generator stuff
     * @param mat: transition matrix
     * @param R: deque to hold random numbers
     * @param M: matrix to represent voter CFTP process
     * @param nStates: how many states?
     * @param coalesced: has the chain coalesced?
     * @return: distribution
     */
    int iteratedVoterCFTP( std::mt19937 &gen, std::uniform_real_distribution<> &dis,
                          const Eigen::MatrixXd &mat, std::deque<double> &R,
                          Eigen::MatrixXd &M, Eigen::MatrixXd &temp,
                          int &nStates, bool coalesced);
    /**
     * @author: Zane Jakobs
     * @param mat: matrix to sample from
    * @param n: how many samples
     * @return: vector where i-th entry is the number of times state i appeared
     */
    valarray<int> sampleVoterCFTP(const Eigen::MatrixXd &mat, int n);

    /**
     * @author: Zane Jakobs
     * @param mat: matrix to sample from
     * @param n: how many samples
     * @return: VectorXd where i-th entry is the density of state i
    */
    Eigen::VectorXd voterCFTPDistribution(Eigen::MatrixXd &mat, int n);
}


#endif /* CFTP_hpp */
