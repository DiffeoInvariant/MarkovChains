/**
 * @summary : implementation of a Markov chain. Basic class template and some functions from
 * https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig.
 * Note that any individual functions not written by the author here have source links in the comments above the declaration
 * @author : Zane Jakobs
 */
#ifndef MarkovChain_hpp
#define MarkovChain_hpp

#include<vector>
#include"Eigen/Core"
#include"Eigen/Eigenvalues"
#include<random>
#include<iostream>
#include<cmath>
using namespace std;

class MarkovChain
{
    
public:
    MarkovChain();
    
    ~MarkovChain();
    //transition and initial probability matrix and vector
    
    
    
    /**
     * @name MarkovChain::setModel
     * @summary : setModel sets parameters
     * @param: transition: N x N transition matrix
     * @param: initial: 1 x N initial probability row vector
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     */
    
    void setModel( Eigen::MatrixXd transition, Eigen::MatrixXd initial, int _numStates){
        _transition = transition;
        _initial = initial;
        numStates = _numStates;
    }
    
    void setTransition(Eigen::MatrixXd transition){
        _transition = transition;
}
    
    void setInitial(Eigen::MatrixXd initial){
        _initial = initial;
    }
    void setNumStates(int _num){
        numStates = _num;
    }
    
    Eigen::MatrixXd getTransition(void) const { return _transition; }
    Eigen::MatrixXd getInit() const { return _initial; }
    int getNumStates(void) const {return numStates;}
    /**
     ** @name MarkovChain::computeProbability
     * @summary: probability that the given sequence was generated
     * by the Markov chain modeled by an instance of this class.
     * @param sequence: a vector of ints, starting at 0
     * @return: p, the probability that that the Markov chain generates
     * this sequence.
     */
    double computeProbability(vector<int> sequence);
    
    
    /**
     * @name MarkovChain::randTransition
     * @summary: initialRand: generates random state
     * @param matrix: transition matrix
     * @param index: current state
     * @return index corresponding to the transition we make
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     * @modifications: changed random number generator to modern standard
     */
    int randTransition(Eigen::MatrixXd matrix, int index);
    
    /**
     * @name MarkovChain::generateSequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @return: vector of ints representing the sequence
     * @ source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     */
    vector<int> generateSequence(int n);
    
    /**
     * @name MarkovChain::stationaryDistributions
     * @summary: stationaryDistributions returns the last stationary distributions of the
     * Markov Chain.
     * @return: vector of doubles corresponding to the last stationary distribution (by order of the eigenvalues)
     */
    Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> stationaryDistributions();
    
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
     * @author: Zane Jakobs
     * @return: matrix of number of paths of length n from state i to state j
     */
    Eigen::MatrixXd numPaths(int n);
        
        /**
         * @author: Zane Jakobs
         * @return 1 or 0 for if state j can be reached from state i
         */
    Eigen::MatrixXd isReachable(void);
        
        /**
         * @author: Zane Jakobs
         * @return matrix where each row has a 1 in the column of each element in that communicating class (one row = one class)
         */
    Eigen::MatrixXd communicatingClasses(void);
    
private:
    Eigen::MatrixXd _transition, _initial;
    int numStates;
};
#endif

