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
#include "MatrixFunctions.hpp"
#include<random>
#include<iostream>
#include<cmath>
using namespace std;

class MarkovChain
{
    
public:
    
    friend std::ostream& operator<<(ostream &os, const MarkovChain &MC){
        os << "Probability Matrix:" <<endl;
        
        Eigen::MatrixXd mat =  MC.getTransition();
        os << "  || ";
        for(int i = 0; i < mat.cols(); i++){
            os << i << " ";
        }
        os <<endl << "  ";
        
        static const int widthMult = 6;
        for(int i = 0; i < widthMult*mat.cols(); i++){
            os << "=";
        }
        os << endl;
        for(int i = 0; i < mat.cols(); i++){
            os << i << " || ";
            for(int j = 0; j < mat.cols(); j++){
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
    
    
    
    
    
    
    MarkovChain() {};
    
    ~MarkovChain(){
        _transition.resize(0,0);
        _initial.resize(0,0);
        numStates = 0;
    }
    //transition and initial probability matrix and vector
    
    
    /**
     * @name MarkovChain::randTransition
     * @summary: initialRand: generates random state
     * @param matrix: transition matrix
     * @param index: current state
     * @return index corresponding to the transition we make
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     * @modifications: changed random number generator to modern standard
     */
    int randTransition(Eigen::MatrixXd matrix, int index){
        //set random seed
        random_device rd;
        //init Mersenne Twistor
        mt19937 gen(rd());
        // unif(0,1)
        uniform_real_distribution<> dis(0.0,1.0);
        double u = dis(gen);
        double s = matrix(index,0);
        int i = 0;
        
        while(u > s && (i < matrix.cols())){
            s += matrix(index,i);
            i += 1;
        }
        
        return i;
    }
    
    /**
     * @name MarkovChain::generateSequence
     * @summary: generateSequence generates a sequence of length n from the Markov chain
     * @param n: length of sequence
     * @return: vector of ints representing the sequence
     * @ source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     */
    vector<int> generateSequence(int n){
        std::vector<int> sequence(n);
        int i, index;
        i = 0;
        index = 0;
        
        //generate initial state
        int init = randTransition(_initial,0);
        sequence[i] = init;
        index = init;
        for(i = 1; i<n; i++){
            index = randTransition(_transition,index);
            sequence[i] = index;
        }
        return sequence;
        
    }
    
    /**
     * @name MarkovChain::stationaryDistributions
     * @summary: stationaryDistributions returns the last stationary distributions of the
     * Markov Chain.
     * @return: vector of doubles corresponding to the last stationary distribution (by order of the eigenvalues)
     */
    Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> stationaryDistributions(){
        //instantiate eigensolver
        Eigen::EigenSolver<Eigen::MatrixXd> es;
        //transpose transition matrix to find left eigenvectors
        Eigen::MatrixXd pt = _transition.transpose();
        //compute evecs, evals
        es.compute( pt, true);
        //store evals
        Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> evals = es.eigenvalues();
        
        Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> evecs = es.eigenvectors();
        Eigen::Matrix<complex<double>,1,2> dotmat;
        dotmat(0,0) = 1;
        dotmat(0,1) = 0;
        int count = 0;
        int index;
        for(int i = 0; i< 2; i++){
            if(abs((evals(i)*dotmat).sum()) <= 1.01 && abs((evals(i)*dotmat).sum()) >= 0.99){
                count++;
                index = i;
            }
        }
        
        std::cout << endl << count << " eigenvalue(s) is (are) 1, and the last stationary distribution, (stateN,0), is " << (evecs.col(index).transpose())/(evecs.col(index).sum()) <<endl;
        return (evecs.col(index).transpose())/(evecs.col(index).sum());
    }
    
    /**
     * @name MarkovChain::limitingDistribution
     * @summary: limitingDistribution computes the limiting distribution of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat.row(0): limiting distribution
     */
    Eigen::MatrixXd limitingDistribution(int expon){
        Eigen::MatrixXd limmat;
        limmat = _transition.pow(expon);
        return limmat.row(0);
    }
    
    /**
     * @name MarkovChain::limitingMat
     * @summary: limitingMat computes the infinite-limit matrix of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat: limiting distribution matrix
     */
    Eigen::MatrixXd limitingMat(int expon){
        Eigen::MatrixXd limmat;
        limmat = _transition.pow(expon);
        return limmat;
    }
    
    /**
     * @summary: does mat contain key?
     * @return: answer to above question, true or false for yes or no
     */
    bool contains(Eigen::MatrixXd mat, double key){
        int n = mat.cols();
        for(int i = 0; i<n; i++){
            for(int j = 0; j < n; j++){
                if(mat(i,j) == key)  return true;
            }
        }
        return false;
    }
    /**
     * @author: Zane Jakobs
     * @return: matrix of number of paths of length n from state i to state j
     */
    Eigen::MatrixXd numPaths(int n){
        Eigen::MatrixXd logicMat(numStates,numStates);
        for(int i = 0; i< numStates; i++){
            for(int j = 0; j < numStates; j++){
                if(_transition(i,j) != 0){
                    logicMat(i,j) = 1;
                }
                else{
                    logicMat(i,j) = 0;
                }
            }//end inner for
        }
        if(n > 1){
            Eigen::MatrixXd powmat = logicMat;
            for(int i = 0; i < n; i++){
                powmat = powmat * logicMat;
            }
            return powmat;
        }
        
        return logicMat;
    }
    
    
    /**
     * @author: Zane Jakobs
     * @return 1 or 0 for if state j can be reached from state i
     */
    Eigen::MatrixXd isReachable(void){
        Eigen::MatrixXd R(numStates,numStates);
        Eigen::MatrixXd Rcomp = Eigen::MatrixXd::Identity(numStates,numStates);
        
        Rcomp = (Rcomp + numPaths(1));
        Eigen::MatrixXd rpow = Rcomp;
        for(int i = 0; i < numStates - 1; i++){
            rpow = rpow * Rcomp;
        }
        for(int i = 0; i< numStates; i++){
            for(int j = 0; j<numStates; j++){
                if(Rcomp(i,j) > 0){
                    R(i,j) = 1;
                }
                else{
                    
                    R(i,j) = 0;
                }
            }
        }
        return R;
    }
    
    
    bool isInVec(std::vector<int> v, int key){
        for(std::vector<int>::iterator it = v.begin(); it != v.end(); it++){
            if( *it == key){
                return true;
            }
        }
        return false;
    }
    /**
     * @author: Zane Jakobs
     * @return matrix where each row has a 1 in the column of each element in that communicating class (one row = one class)
     */
    Eigen::MatrixXd communicatingClasses(void){
        Eigen::MatrixXd CC(numStates,numStates);
        Eigen::MatrixXd reachable = isReachable();
        for(int i = 0; i< numStates; i++){
            for(int j = 0; j < numStates; j++){
                if(reachable(i,j) == 1 && reachable(j,i) == 1){
                    CC(i,j) = 1;
                }
                else{
                    CC(i,j) = 0;
                }
            }//end inner for
        }//end outer for
        return CC;
        
        
        int count = 0; //how many classes?
        bool found;
        Eigen::MatrixXd uniqueC(1,numStates);
        for(int i = 0; i< numStates; i++ ){
            found = false;
            for(int j = i +1; j < numStates; j++){
                if(CC.row(j) == CC.row(i)){
                    found = true;
                }
            }
            if(!found){
                if(count > 0){
                    uniqueC.conservativeResize(uniqueC.rows() +1, uniqueC.cols());
                }
                uniqueC.row(count) = CC.row(i);
                count++;
            }
        }//end outer for
        
        return uniqueC;
        
    }
    
private:
    Eigen::MatrixXd _transition, _initial;
    int numStates;
};
#endif

