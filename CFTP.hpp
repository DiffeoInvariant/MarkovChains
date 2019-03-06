#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#ifndef CFTP_hpp
#define CFTP_hpp

#include "Eigen/Core"
#include<deque>
#include<valarray>
#include<random>
#include<algorithm>
#include<type_traits>
#include<mkl.h>
using namespace std;
typedef std::complex<double> CD;
typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixXcd;
/**
 * Coupling From The Past to perfectly sample from a Markov chain matrix mat
 */
namespace Markov
{
/** template checks if object T has member _transition
 *
 */
    /*
#define DEFINE_MEMBER_CHECKER(member) \
    template<typename T, typename V = bool> \
    struct has_ ## member : std::false_type{}; \
    template<typename T> \
    struct has_ ## member<T, typename enable_if< \
    !is_same<decltype(declval<T>().member), void>::value, bool >::type > : true_type {};
    
#define HAS_MEMBER(C,member) \
    has_ ## member<C>::value
     */
    
    

Eigen::MatrixXd matPow(Eigen::MatrixXd &mat, int _pow);
 
    
    /**
     * Taken from https://software.intel.com/en-us/node/521147
     * @summary: C++ declaration of FORTRAN function dgeev
     *
     */
    extern "C" lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda, double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr );
//void *threadedMatPow(Eigen::MatrixXd &mat, int pow);
    
    /**
     * taken from print function in https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgeev_col.c.htm
     fills matrix where columns are eigenvectors in row major order
     * @param n: dimension of matrix
     * @param v: array of eigenvectors
     * @param ldv: dimension of array v
     */
    Eigen::MatrixXcd LAPACKE_evec_to_Eigen(MKL_INT n, double* wi, double* v, MKL_INT ldv);
    
    /**
     * taken from print function in https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgeev_col.c.htm
     fills vector with eigenvalues
     */
    Eigen::MatrixXcd LAPACKE_eval_to_Eigen(MKL_INT n, double* wr, double* wi);
    
    /**
     * @summary: solves eigen-problem
     * A * v(i) = lambda(i)* v(i)
     * @return: true for success, false for failure
     */
    bool eigenProblem(Eigen::MatrixXd& A, MatrixXcd& v,
                      MatrixXcd& lambda);
/**
 * @summary: solves generalized eigen-problem:
 * A * v(i) = lambda(i) * B* v(i)
 * taken from http://eigen.tuxfamily.org/index.php?title=Lapack
 */
    bool eigenProblem(MatrixXcd& A, MatrixXcd& v, MatrixXcd& lambda);
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
