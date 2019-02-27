
#ifndef CFTP_hpp
#define CFTP_hpp

#include "Eigen/Core"
#include<deque>
#include<valarray>
#include<random>
#include<algorithm>
#include<type_traits>
using namespace std;
/**
 * Coupling From The Past to perfectly sample from a Markov chain matrix mat
 */
namespace Markov
{
/** template checks if object T has member _transition
 *
 */
#define DEFINE_MEMBER_CHECKER(member) \
    template<typename T, typename V = bool> \
    struct has_ ## member : std::false_type{}; \
    template<typename T> \
    struct has_ ## member<T, typename enable_if< \
        !is_same<decltype(declval<T>().member), void>::value, bool >::type > : true_type {};
    
#define HAS_MEMBER(C,member) \
    has_ ## member<C>::value
Eigen::MatrixXd matPow(Eigen::MatrixXd &mat, int _pow);
 
//void *threadedMatPow(Eigen::MatrixXd &mat, int pow);

double variation_distance(Eigen::MatrixXd dist1, Eigen::MatrixXd dist2);

double k_stationary_variation_distance(Eigen::MatrixXd trans, int k);
int mixing_time(Eigen::MatrixXd &trans);

int isCoalesced(Eigen::MatrixXd &mat);

int random_transition(Eigen::MatrixXd &mat, int init_state, double r);

/**
 * @author: Zane Jakobs
 * @summary: voter CFTP algorithm to perfectly sample from the Markov chain with transition matrix mat. Algorithm from https://pdfs.semanticscholar.org/ef02/fd2d2b4d0a914eba5e4270be4161bcae8f81.pdf
 * @return: perfect sample from matrix's distribution
 */
int voter_CFTP(Eigen::MatrixXd &mat);

int iteratedVoterCFTP( std::mt19937 &gen, std::uniform_real_distribution<> &dis, Eigen::MatrixXd &mat, std::deque<double> &R, Eigen::MatrixXd &M, Eigen::MatrixXd &temp, int &nStates, bool coalesced);
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
Eigen::VectorXd voterCFTPDistribution(Eigen::MatrixXd &mat, int n);
}


#endif /* CFTP_hpp */
