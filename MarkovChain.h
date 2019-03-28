/**
 * @summary : implementation of a Markov chain.
 * Note that any individual functions not written by the author here have source links in the comments above the declaration
 * @author : Zane Jakobs
 */
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#ifndef MarkovChain_hpp
#define MarkovChain_hpp

#include<Eigen/Core>
#include<Eigen/Dense>
#include<mkl.h>
#include<type_traits>
#include<complex>
#include"MarkovFunctions.hpp"
using namespace std;
using namespace Markov;
namespace Markov
{
    

class MarkovChain
{
    
protected:
    unsigned numStates;
    Eigen::MatrixXd _transition, _initial;
    
public:
    
    friend ostream& operator<<(ostream &os, const MarkovChain &MC){
        os << "Probability Matrix:" <<endl;
        auto ns = MC.getNumStates();
        Eigen::MatrixXd mat =  MC.getTransition();
        os << "  || ";
        for(int i = 0; i < ns; i++){
            os << i << " ";
        }
        os <<endl << "  ";
        
        int widthMult = 6;
        for(int i = 0; i < widthMult*ns; i++){
            os << "=";
        }
        os << endl;
        for(int i = 0; i < ns; i++){
            os << i << " || ";
            for(int j = 0; j < ns; j++){
                os << mat(i,j) << " , ";
            }
            os << "||" <<endl;
        }
        return os;
    }
    
    
    /**
     * @name MarkovChain::setModel
     * @summary : setModel sets parameters
     * @param: transition: N x N transition matrix
     * @param: initial: 1 x N initial probability row vector
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     */
    
    void setModel( Eigen::MatrixXd& transition, Eigen::MatrixXd& initial, int _numStates);
    
    void setTransition(Eigen::MatrixXd& transition);
    void setInitial(Eigen::MatrixXd& initial);
    void setNumStates(int _num);
    
    Eigen::MatrixXd getTransition(void) const;
    Eigen::MatrixXd getInit() const;
    int getNumStates(void) const;
    
    MarkovChain() {}
    
    MarkovChain( Eigen::MatrixXd transition, Eigen::MatrixXd initial, int _numStates){
        _transition = transition;
        _initial = initial;
        numStates = _numStates;
    }
    ~MarkovChain(){
        //_transition.resize(0,0);
       // _initial.resize(0,0);
        ~_transition;
        ~_initial;
        numStates = 0;
    }
    
    
    
    
    /**
     * @author: Zane Jakobs
     * @summary: counts transitions
     * @param dat: the data
     * @param oversize: upper bound on the number of states in the chain
     * @return: estimate of transition matrix, with zero entries where they should be to
     * preserve communicating classes
     */
    static auto countMat(vector<Sequence> df, int oversize = 100);
    /**
     * @author: Zane Jakobs
     * @summary: computes transition matrix based on empirical distribution of values. In
     * particular, attempts to find which entries should be zero
     * @param dat: the data
     * @param oversize: upper bound on the number of states in the chain
     * @return: maximum likelihood estimator of _transition
     */
    static auto MLE(vector<Sequence> df, int oversize = 100);
    
    /**
     * @name MarkovChain::generateSequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @return: vector of ints representing the sequence
     */
    auto generateSequence(int n);
    /**
     * @name MarkovChain::stationaryDistributions
     * @summary: stationaryDistributions returns the last stationary distributions of the
     * Markov Chain.
     * @return: vector of doubles corresponding to the last stationary distribution (by order of the eigenvalues)
     */
    Eigen::MatrixXcd stationaryDistributions();
    
    /**
     * @name MarkovChain::limitingDistribution
     * @summary: limitingDistribution computes the limiting distribution of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat.row(0): limiting distribution
     */
    Eigen::MatrixXd limitingDistribution(int expon);
    
    /**
     * @name MarkovChain::limitingMat
     * @summary: limitingMat computes the infinite-limit matrix of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat: limiting distribution matrix
     */
    Eigen::MatrixXd limitingMat(int expon);
    
    /**
     * @summary: does mat contain key?
     * @return: answer to above question, true or false for yes or no
     */
    bool contains(Eigen::MatrixXd mat, double key);
    /**
     * @author: Zane Jakobs
     * @return: matrix of number of paths of length n from state i to state j
     */
    Eigen::MatrixXd numPaths(int n);
    
    /**
     * @author: Zane Jakobs
     * @return 1 or 0 for if state j can be reached from state i
     */
    auto isReachable();
    
    template<typename T>
    constexpr bool isInVec(std::vector<T> v, T key);
    /**
     * @author: Zane Jakobs
     * @return matrix where each row has a 1 in the column of each element in that communicating class (one row = one class)
     */
    Eigen::MatrixXd communicatingClasses();
 
    
    /**
     * @author: Zane Jakobs
     * @return: expected value of (T = min n >= 0 s.t. X_n  = sh) | X_0 = s0
     * @param s0: initial state
     * @param sh: target state
     */
    auto expectedHittingTime(int s0, int sh);
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @param sEnd: end state
     * @return: mean time the chain spends in sInt, starting at s0, before returning to s0
     */
    auto meanTimeInStateBeforeReturn(int s0, int sInt);
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @param sEnd: end state
     * @return: mean time the chain spends in sInt, starting at s0, before hitting sEnd
     */
    auto meanTimeInStateBeforeHit(int s0, int sInt, int sEnd);
    
    /**
     * @author: Zane Jakobs
     * @param t: time difference
     * @return: cov(X_s,X_{s+t}), taken from HMM for Time Series: an Intro Using R page 18
     */
    auto cov(int t);
    

};
    
}

#endif

