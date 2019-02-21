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
#include"Eigen/Dense"
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
        numStates = transition.cols();
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
    static int randTransition(Eigen::MatrixXd matrix, int index){
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
                if(rpow(i,j) > 0){
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
        int count = 0; //how many classes?
        bool found;
        Eigen::MatrixXd uniqueC(1,numStates);
        for(int i = 0; i< numStates; i++ ){
            found = false;
            for(int j = 0; j < numStates; j++){
                if(CC.row(j) == CC.row(i) && j != i){
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
    /**
     * @author: Zane Jakobs
     * @return: expected value of (T = min n >= 0 s.t. X_n  = sh) | X_0 = s0
     * @param s0: initial state
     * @param sh: target state
     */
    double expectedHittingTime(int s0, int sh){
        /**
         Algorithm: let u_i = E[T|X_0 = i]. Set up system of equations. Solve for u_s0. We have the equation Au = c, where u = (u_0, u_1, ..., u_n)^T, c = (-1,-1,...,0,-1,...-1)^T, with 0 in the spot corresponding to sh, and
         
                            [p00-1, p01, p02, ... p0n]
                            [p10, p11-1, p12, ... p1n]
                    A =     [p20, p21, p22-1, ... p1n]
                                      ...
                            [0, 0, 0, ... 1, 0, ... 0]
                            [p(sh+1)0, ..., p(sh+1)n]
                            [pn0-1, pn1, pn2, ... pnn-1]
         
         where the 1 in the sh-th row is in the sh position.
         */
        Eigen::MatrixXd A(numStates, numStates);
        Eigen::VectorXd u(numStates);
        Eigen::VectorXd c(numStates);
        
        for(int i = 0; i<numStates; i++){
            //fill c
            if(i == sh){
                c(i) = 0;
            }
            else{
                c(i) = -1;
            }
        //fill A
        for(int j = 0; j< numStates; j++){
            //if we're in the row of 0's and a 1
            if(i == sh){
                if(j == sh){
                    A(i,j) = 1;
                }
                else{
                    A(i,j) = 0;
                }
            }//end if
            else{
                (j == i) ? (A(i,j) = _transition(i,j)-1) : (A(i,j) == _transition(i,j));
            }
        }//end inner for
    }//end outer for
    
    /*choose linear solver based on matrix size. Using info from Eigen docs at
     https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
     */
        if(numStates < 500){
            //for smaller matrices, rank-revealing Householder QR decomposition with column pivoting
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> CPH(A);
            u = CPH.solve(c);

        }
        else{
            //for big ones, use regular Householder QR
            Eigen::HouseholderQR<Eigen::MatrixXd> HQR(A);
            u = HQR.solve(c);
        }
        const double err_tol = 1.0e-4; //error tolerance on solutions
        double relative_error = (A*u - c).norm() / c.norm();
        if(relative_error < err_tol){
            return u(s0);
        }
        else{
            return -1; //return -1 if not in error bound
        }
    }
    
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @param sEnd: end state
     * @return: mean time the chain spends in sInt, starting at s0, before returning to s0
     */
    double meanTimeInStateBeforeReturn(int s0, int sInt){
        /*
         Algorithm: We want to solve Aw = c, where c = (0,0,...,-1,0,...,0)^T, with the -1 in the sInt position, and w = (w0, w1, ..., wsInt, ... ,wn)^T. We want ws0. We also define
         
                [p00-1, p01, ..., 0, ...,p0n]
                [p10, p11-1, ..., 0, ...,p1n]
         A =    [p20, p21, ..., 0, ...,  p2n]
                            ...
                [p(sEnd+1)0, p(sEnd+1)1, ...]
                [pn0, pn1, ..., 0, ..., pnn-1]
         
         where the 0's are in the s0 column
         */
        Eigen::MatrixXd A(numStates, numStates);
        Eigen::VectorXd w(numStates);
        Eigen::VectorXd c(numStates);
        
        for(int i = 0; i<numStates; i++){
            //fill c
            if(i == sInt){
                c(i) = -1;
            }
            else{
                c(i) = 0;
            }
            //fill A
            for(int j = 0; j< numStates; j++){
                //if we're in column s0
                if(j == s0){
                    (i == s0) ? A(i,j) = -1 : A(i,j) = 0;
                }
                else{
                    (j == i) ? A(i,j) = _transition(i,j) -1 : A(i,j) = _transition(i,j);
                }
            }//end inner for
        }//end outer for
        
        /*choose linear solver based on matrix size. Using info from Eigen docs at
         https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
         */
        if(numStates < 500){
            //for smaller matrices, rank-revealing Householder QR decomposition with column pivoting
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> CPH(A);
            w = CPH.solve(c);
        }
        else{
            //for big ones, use regular Householder QR
            Eigen::HouseholderQR<Eigen::MatrixXd> HQR(A);
            w = HQR.solve(c);
        }
        const double err_tol = 1.0e-4; //error tolerance on solutions
        double relative_error = (A*w - c).norm() / c.norm();
        if(relative_error < err_tol){
            return w(s0);
        }
        else{
            return -1; //return -1 if not in error bound
        }
    }
    /**
     * @author: Zane Jakobs
     * @param s0: initial state
     * @param sInt: intermediate state
     * @param sEnd: end state
     * @return: mean time the chain spends in sInt, starting at s0, before hitting sEnd
     */
    double meanTimeInStateBeforeHit(int s0, int sInt, int sEnd){
        /*
         Algorithm: We want to solve Aw = c, where c = (0,0,...,-1,0,...,0)^T, with the -1 in the sInt position, and w = (w0, w1, ..., wsInt, ... ,wn)^T. We want ws0. We also define
         
                [p00-1, p01, ..., 0, ...,p0n]
                [p10, p11-1, ..., 0, ...,p1n]
         A =    [p20, p21, ..., 0, ...,  p2n]
                            ...
                [0, 0, 0, ... , 0, ..., 0, 0]
                [p(sEnd+1)0, p(sEnd+1)1, ...]
                [pn0, pn1, ..., 0, ..., pnn-1]
         
         where the 0's are in the sEnd row and column
         */
        Eigen::MatrixXd A(numStates, numStates);
        Eigen::VectorXd c(numStates);
        Eigen::VectorXd w(numStates);
        //initialize c
        for(int i = 0; i < numStates; i++){
            if(i == sInt){
                c(i) = -1;
            }
            else{
                c(i) = 0;
            }
        }//end for
        
        //initialize A
        for(int i = 0; i< numStates; i++){
            if(i == sEnd){
                for(int j = 0; j< numStates; j++){
                    A(i,j) = 0;
                }
            }//end if
            else{
                for(int j = 0; j< numStates; j++){
                    (j== sEnd) ? A(i,j) = 0 : A(i,j) = _transition(i,j);
                    if(j == i){
                        A(i,j) -= 1.0;
                    }
                }
            }//else
        }//end outer for
        
        /*choose linear solver based on matrix size. Using info from Eigen docs at
         https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
         */
        if(numStates < 500){
            //for smaller matrices, rank-revealing Householder QR decomposition with column pivoting
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> CPH(A);
            w = CPH.solve(c);
        }
        else{
            //for big ones, use regular Householder QR
            Eigen::HouseholderQR<Eigen::MatrixXd> HQR(A);
            w = HQR.solve(c);
            
        }
        const double err_tol = 1.0e-4; //error tolerance on solutions
        double relative_error = (A*w - c).norm() / c.norm();
        if(relative_error < err_tol){
            return w(s0);
        }
        else{
            return -1; //return -1 if not in error bound
        }
        
        
    }//end function
    
    
private:
    Eigen::MatrixXd _transition, _initial;
    int numStates;
};
#endif

