/**
 * @summary : implementation of a Markov chain.
 * @author : Zane Jakobs
 */
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef MarkovChain_h
#define MarkovChain_h

class MarkovChain;
#ifdef Success
#undef Success
#endif
#include<Eigen/Core>
#include <Eigen/QR>
#include<Eigen/Dense>
#include<mkl.h>
#include<type_traits>
#include<complex>
#include"MarkovFunctions.h"
using namespace std;
using namespace Markov;
namespace Markov
{
    /**
     wrapper around std::vector<int>
     */
    typedef struct
    {
        std::vector<int> seq;
        void set_seq(std::vector<int> _seq){
            seq = _seq;
        }
    }Sequence;

class MarkovChain
{
    
protected:
    unsigned numStates;
    Eigen::MatrixXd _transition, _initial;
    
public:
    
    friend ostream& operator<<(ostream &os, const MarkovChain &MC){
        os << "Probability Matrix:" <<"\n";
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
            os << "||" << "\n";
        }
        return os;
    }
    
    
    /**
     * @name MarkovChain::setModel
     * @summary : setModel sets parameters
     * @param: transition: N x N transition matrix
     * @param: initial: 1 x N initial probability row vector
     */
    
    void setModel( const Eigen::MatrixXd& transition, const Eigen::MatrixXd& initial, int _numStates);
    
    void setTransition(const Eigen::MatrixXd& transition);
    void setInitial(const Eigen::MatrixXd& initial);
    void setNumStates(int _num);
    
    Eigen::MatrixXd getTransition() const;
    Eigen::MatrixXd getInit() const;
    int getNumStates() const;
    
    MarkovChain() {}
    
    MarkovChain( const Eigen::MatrixXd& transition, const Eigen::MatrixXd& initial, int _numStates){
        _transition = transition;
        _initial = initial;
        numStates = _numStates;
    }
    ~MarkovChain(){
        _transition.resize(0,0);
        _initial.resize(0,0);
        numStates = 0;
    }

    /**
     * @author: Zane Jakobs
     * @summary: computes transition matrix based on empirical distribution of values. In
     * particular, attempts to find which entries should be zero
     * @param df: the data
     * @param oversize: upper bound on the number of states in the chain
     * @return: maximum likelihood estimator of _transition
     */
    static Eigen::MatrixXd MLE(const vector<Sequence>& df, int oversize = 100);
    
    /**
     * @name MarkovChain::generateSequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @return: vector of ints representing the sequence
     */
    vector<int> generateSequence(int n) const noexcept;
    /**
     * @name MarkovChain::stationaryDistributions
     * @summary: stationaryDistributions returns the last stationary distributions of the
     * Markov Chain.
     * @return: vector of doubles corresponding to the last stationary distribution (by order of the eigenvalues)
     */
    Eigen::MatrixXcd stationaryDistributions() const;
    
    /**
     * @name MarkovChain::limitingDistribution
     * @summary: limitingDistribution computes the limiting distribution of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat.row(0): limiting distribution
     */
    Eigen::MatrixXd limitingDistribution(int expon) const;
    
    /**
     * @name MarkovChain::limitingMat
     * @summary: limitingMat computes the infinite-limit matrix of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat: limiting distribution matrix
     */
    Eigen::MatrixXd limitingMat(int expon) const;
    
    /**
     * @summary: does mat contain key?
     * @return: answer to above question, true or false for yes or no
     */
    bool contains(const Eigen::MatrixXd& mat, double key) const noexcept;
    /**
     * @author: Zane Jakobs
     * @return: matrix of number of paths of length n from state i to state j
     */
    Eigen::MatrixXd numPaths(int n) const;
    
    /**
     * @author: Zane Jakobs
     * @return 1 or 0 for if state j can be reached from state i
     */
    Eigen::MatrixXd isReachable() const;
    
    template<typename T>
    constexpr static bool isInVec(const std::vector<T>& v, T key);
    /**
     * @author: Zane Jakobs
     * @return matrix where each row has a 1 in the column of each element in that communicating class (one row = one class)
     */
    Eigen::MatrixXd communicatingClasses() const;
 
    
    /**
     * @author: Zane Jakobs
     * @return: expected value of (T = min n >= 0 s.t. X_n  = sh) | X_0 = s0
     * @param s0: initial state
     * @param sh: target state
     */
    double expectedHittingTime(int s0, int sh) const;
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @return: mean time the chain spends in sInt, starting at s0, before returning to s0
     */
    double meanTimeInStateBeforeReturn(int s0, int sInt) const;
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @param sEnd: end state
     * @return: mean time the chain spends in sInt, starting at s0, before hitting sEnd
     */
    double meanTimeInStateBeforeHit(int s0, int sInt, int sEnd) const;
    
    /**
     * @author: Zane Jakobs
     * @param t: time difference
     * @return: cov(X_s,X_{s+t}), taken from HMM for Time Series: an Intro Using R page 18
     */
    double cov(int t) const;
    /**
     * @author: Zane Jakobs
     * @param t: time difference
     * @return: corr(X_s,X_{s+t}), taken from HMM for Time Series: an Intro Using R page 18
     */
    double corr(int t) const;
    
    /**
     *@author Zane Jakobs
     *@return log likelihood of the MLE for a given dataset
     */
    double log_likelihood(const vector<Sequence>& df, int oversize) const;
};
    
}

#endif

