
#ifndef MarkovFunctions_hpp
#define MarkovFunctions_hpp
#include<mkl.h>
#include<vector>
#include<Eigen/Core>
#include<random>
#include<iostream>
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
    constexpr auto make_mapply_pair(T obj, UnaryOp fun)
    {
        std::pair<T, UnaryOp> mpair(obj, fun);
        return mpair;
    }

    /**
     implementing piping for univariate functions; up here is piping with pass by value (lotsa
     */
    template<class P>//pipe closure
    struct pipe_closure : P
    {
        template<class... Xs>
        pipe_closure(Xs&&... xs) : P(std::forward<Xs>(xs)...) {}
    };
    
    template<class T, class P>
    decltype(auto) operator|(T&& x, const pipe_closure<P>& p)
    {
        return p(std::forward<T>(x));
    }
    struct add_one_f //sample function to see if this shit works
    {
        template<class T>
        auto operator()(T x) const
        {
            return x + 1;
        }
    };
    struct real_matrix_max_f
    {
        //T must be an Eigen matrix with 1 or more rows and columns
      template<class T>
        auto operator()(T x) const
        {
            auto max = x(0,0);
            int rows = x.rows();
            int cols = x.cols();
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                    if(!( i == 0 && j == 0)){
                        if(x(i,j) > max){
                            max = x(i,j);
                        }//end if
                    }//end if
                }//end inner for
            }//end outer for
            return max;
        }
    };
    
  //T must be an eigen matrix and UnaryOp a functor
    struct mat_apply_f
    {
        template<typename T> //T is a std pair
        auto operator()(T mpair) const
        {
            int rows = mpair.first.rows();
            int cols = mpair.first.cols();
            auto newmat = mpair.first;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                    newmat(i,j) = mpair.second(mpair.first(i,j));
                }//end inner for
            }//end outer for
            return newmat;
        }
    };
    //wrapper for perfect forwarding (that is, passing the reference on through
    //the chain
    #define REQUIRES(...) class=std::enable_if_t<(__VA_ARGS__)>
    template<class T>
    struct wrapper
    {
        T value;
        template<class X, REQUIRES(std::is_convertible<T,X>())>
        wrapper(X&& x) : value(std::forward<X>(x)) {}
        
        T get() const
        {
            return std::move(value);
        }
    };
    
    template<class T>
    auto make_wrapper(T&& x)
    {
        return wrapper<T>(std::forward<T>(x));
    }
    
    template<class P>
    auto make_pipe_closure(P p){
        return pipe_closure<P>(std::move(p));
    }
    template<class P>
    struct pipable
    {
        template<class... Ts>
        auto operator()(Ts&&... xs) const
        {
            return make_pipe_closure([](auto... ws)
            {
                return [=](auto&& x) -> decltype(auto)
                {
                    return P()(x,ws.get()...);
                };
            }(make_wrapper(std::forward<Ts>(xs)...)));
        }
    };
    
    template<class T, class P>
    decltype(auto) operator|(T&& x, const pipable<P>& p)
    {
        return P()(std::forward<T>(x));
    }
    

/**
 * Taken from https://software.intel.com/en-us/node/521147
 * @summary: C++ declaration of FORTRAN function dgeev
 *
 */
    extern "C" lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda, double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr );


    typedef std::complex<double> CD;

    typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixXcd;

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
    bool eigen_problem(Eigen::MatrixXd& A, MatrixXcd& v,
                   MatrixXcd& lambda);

    Eigen::MatrixXd normalize_rows(Eigen::MatrixXd &mat);
    /**
     *@author: Zane Jakobs
     *@param mat: matrix to raise to power, passed by value for safe use with class members
     *@param expon: power
     *@return: mat^expon
     */
    Eigen::MatrixXd matrix_power(Eigen::MatrixXd& mat, const int& expon);
    
    /**
     * @author: Zane Jakobs
     * @param M: type of thing we're taking the polynomial of--specialized for
     * Eigen::MatrixXd and MatrixXcd. Default template works for numeric data types
     * @return: P(x) = (x-lambda_1)*(x-lambda_2)*...*(x-lambda_n)
     */
    template<typename M>
    auto characteristic_polynomial(Eigen::MatrixXd &mat, M &x);
    
    /**
     * @name MarkovChain::randTransition
     * @summary: initialRand: generates random state
     * @param matrix: transition matrix
     * @param index: current state
     * @param u: random uniform between 0 and 1
     * @return index corresponding to the transition we make
     */
    constexpr auto random_transition(const Eigen::MatrixXd &mat, int nStates, int init_state, double r) noexcept;
    /**
     * @name MarkovChain::generate_mc_sequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @param matT: transition matrix
     * @param initialDist: initial distribution
     * @return: vector of ints representing the sequence
     */
    auto generate_mc_sequence(int n, const Eigen::MatrixXd& matT, const Eigen::MatrixXd& initialDist) noexcept;
}
#endif /* MarkovFunctions_hpp */
