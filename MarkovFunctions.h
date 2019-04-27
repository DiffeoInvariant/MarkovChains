
#ifndef MarkovFunctions_hpp
#define MarkovFunctions_hpp
#include<mkl.h>
#include<vector>
#ifdef Success
#undef Success
#endif
#include<Eigen/Core>
#include<random>
#include<iostream>
#include<cstdio>
#include<cmath>
#include<type_traits>
#include<typeinfo>
#include<complex>
#include<Eigen/LU>
#include<utility>
#include<type_traits>
using namespace std;
using namespace Eigen;
namespace Markov
{
    //compile-time factorial taking advantage of SFINAE
    template<unsigned n>
    struct TMP_factorial
    {
        enum {value = n*TMP_factorial<n-1>::value };
    };
    template<>
    struct TMP_factorial<0>
    {
        enum { value = 1 };
    };
    
    /**
     Taken from Effective Modern C++
     *@author: Scott Meyers
     *@param T: type of array
     *@param N: length of array
     *@return: length of array
     */
    template<typename T, size_t N>
    constexpr size_t arr_size(T (&)[N]) noexcept
    {
        return N;
    }
   
    template<class T, class UnaryOp>
    constexpr decltype(auto) make_mapply_pair(T obj, UnaryOp fun) noexcept
    {
        std::pair<T, UnaryOp> mpair(obj, fun);
        return mpair;
    }


    /**
     * Taken from https://software.intel.com/en-us/node/521147
     * @summary: C++ declaration of FORTRAN function dgeev
     *
     */
    extern "C" lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda, double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr );



    /**
     * @author: Zane Jakobs
     * @param mat: matrix to convert to LAPACKE form
     * @return: pointer to array containing contents of mat in column-major order
     */
    constexpr double* Eigen_to_LAPACKE(const Eigen::MatrixXd& mat);
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
     * @param A: nxn matrix whose eigenstuff we want
     * @param v: nxn matrix to hold eigenvectors
     * @param lambda: 1xn matrix (row vector)
     * @return: true for success, false for failure
     */
    bool eigen_problem(const Eigen::MatrixXd& A, MatrixXcd& v,
                   MatrixXcd& lambda);

    Eigen::MatrixXd normalize_rows(Eigen::MatrixXd &mat);
    /**
     *@author: Zane Jakobs
     *@param mat: matrix to raise to power, passed by value for safe use with class members
     *@param expon: power
     *@return: mat^expon
     */
    Eigen::MatrixXd matrix_power(const Eigen::MatrixXd& mat, const int& expon);
    
    /**
     * @author: Zane Jakobs
     * @param M: type of thing we're taking the polynomial of--specialized for
     * Eigen::MatrixXd and MatrixXcd. Default template works for numeric data types
     * @return: P(x) = (x-lambda_1)*(x-lambda_2)*...*(x-lambda_n)
     */
    template<typename M>
    M characteristic_polynomial(Eigen::MatrixXd &mat, M &x);
    
    /**
     * @name MarkovChain::randTransition
     * @summary: initialRand: generates random state
     * @param matrix: transition matrix
     * @param index: current state
     * @param u: random uniform between 0 and 1
     * @return index corresponding to the transition we make
     */
    constexpr int random_transition(const Eigen::MatrixXd &mat, int nStates, int init_state, double r) noexcept;
    /**
     * @name MarkovChain::generate_mc_sequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @param matT: transition matrix
     * @param initialDist: initial distribution
     * @return: vector of ints representing the sequence
     */
    vector<int> generate_mc_sequence(int n, const Eigen::MatrixXd& matT, const Eigen::MatrixXd& initialDist) noexcept;
}
#endif /* MarkovFunctions_hpp */
