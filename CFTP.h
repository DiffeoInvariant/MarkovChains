#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#ifndef CFTP_hpp
#define CFTP_hpp

#include <Eigen/Core>
#include<deque>
#include<valarray>
#include<random>
#include<algorithm>
#include<type_traits>
#include"MarkovChain.hpp"
#include<mkl.h>
#include<complex>
using namespace std;
typedef std::complex<double>  CD;
typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixXcd;
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
    constexpr auto small_mat_pow(Eigen::MatrixXd &mat, int _pow);
 
   
//void *threadedMatPow(Eigen::MatrixXd &mat, int pow);
   
auto variation_distance(Eigen::MatrixXd dist1, Eigen::MatrixXd dist2);

auto k_stationary_variation_distance(Eigen::MatrixXd trans, int k);
auto mixing_time(Eigen::MatrixXd &trans);

auto isCoalesced(Eigen::MatrixXd &mat);

auto random_transition(Eigen::MatrixXd &mat, int init_state, double r);

/**
 * @author: Zane Jakobs
 * @summary: voter CFTP algorithm to perfectly sample from the Markov chain with transition matrix mat. Algorithm from https://pdfs.semanticscholar.org/ef02/fd2d2b4d0a914eba5e4270be4161bcae8f81.pdf
 * @return: perfect sample from matrix's distribution
 */
auto voter_CFTP(Eigen::MatrixXd &mat);

auto iteratedVoterCFTP( std::mt19937 &gen, std::uniform_real_distribution<> &dis, Eigen::MatrixXd &mat, std::deque<double> &R, Eigen::MatrixXd &M, Eigen::MatrixXd &temp, int &nStates, bool coalesced);
/**
 * @author: Zane Jakobs
 * @param mat: matrix to sample from
 * @param n: how many samples
 * @return: vector where i-th entry is the number of times state i appeared
 */
valarray<int> sampleVoterCFTP(Eigen::MatrixXd &mat, int n);

/**
 * @author: Zane Jakobs
 * @param mat: matrix to sample from
 * @param n: how many samples
 * @return: VectorXd where i-th entry is the density of state i
 */
auto voterCFTPDistribution(Eigen::MatrixXd &mat, int n);
}


#endif /* CFTP_hpp */