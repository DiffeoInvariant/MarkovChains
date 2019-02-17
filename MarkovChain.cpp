/**
* @summary : implementation of a Markov chain. Basic class template and some functions from
* https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig.
* Note that any individual functions not written by the author here have source links in the comments above the declaration
* @author : Zane Jakobs
*/

#include<vector>
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include"MarkovChain.hpp"
#include<random>
#include<iostream>
#include<algorithm>
#include<cmath>
using namespace std;
using namespace Eigen;

ostream& operator<<(ostream &os, MarkovChain &MC){
    os << "Probability Matrix:" <<endl;
    
    Eigen::MatrixXf mat =  MC.getTransition();
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



MarkovChain::MarkovChain() {};

MarkovChain::~MarkovChain(){
        _transition.resize(0,0);
        _initial.resize(0,0);
        numStates = 0;
    }
	//transition and initial probability matrix and vector

    

	/**
    * @name MarkovChain::setModel
	* @summary : setModel sets parameters
    * @param: transition: N x N transition matrix
    * @param: initial: 1 x N initial probability row vector
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
	*/

	/**
    ** @name MarkovChain::computeProbability
	* @summary: probability that the given sequence was generated
	* by the Markov chain modeled by an instance of this class.
	* @param sequence: a vector of ints, starting at 0
	* @return: p, the probability that that the Markov chain generates 
	* this sequence.
	*/
	float MarkovChain::computeProbability(vector<int> sequence){
		float p = 0;
		float init = _initial(0,sequence[0]);
		p = init;

        for(int i = 0; i< sequence.size() -1; i++){
            p *= _transition(sequence[i],sequence[i+1]);
		}

		return p;
	}


	/**
     * @name MarkovChain::randTransition
     * @summary: initialRand: generates random state
     * @param matrix: transition matrix
     * @param index: current state
     * @return index corresponding to the transition we make
     * @source: https://www.codeproject.com/Articles/808292/Markov-chain-implementation-in-Cplusplus-using-Eig
     * @modifications: changed random number generator to modern standard
	*/
    int MarkovChain::randTransition(Eigen::MatrixXf matrix, int index){
		//set random seed
		random_device rd;
		//init Mersenne Twistor
		mt19937 gen(rd());
		// unif(0,1)
		uniform_real_distribution<> dis(0.0,1.0);
		double u = dis(gen);
        float s = matrix(index,0);
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
 	vector<int> MarkovChain::generateSequence(int n){
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
     * @return: vector of floats corresponding to the last stationary distribution (by order of the eigenvalues)
     */
    Eigen::Matrix<complex<float>,Eigen::Dynamic,Eigen::Dynamic> MarkovChain::stationaryDistributions(){
        //instantiate eigensolver
        Eigen::EigenSolver<Eigen::MatrixXf> es;
        //transpose transition matrix to find left eigenvectors
        Eigen::MatrixXf pt = _transition.transpose();
        //compute evecs, evals
        es.compute( pt, true);
        //store evals
        Eigen::Matrix<complex<float>,Eigen::Dynamic,Eigen::Dynamic> evals = es.eigenvalues();
        
        Eigen::Matrix<complex<float>,Eigen::Dynamic,Eigen::Dynamic> evecs = es.eigenvectors();
        Eigen::Matrix<complex<float>,1,2> dotmat;
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
    Eigen::MatrixXf MarkovChain::limitingDistribution(int expon){
        Eigen::MatrixXf limmat = _transition;
        
        for(int i = 1; i<= pow(10,expon); i++){
            limmat = limmat*_transition;
        }
        return limmat.row(0);
    }
    
    /**
     * @name MarkovChain::limitingMat
     * @summary: limitingMat computes the infinite-limit matrix of the Markov chain
     * @param expon: computes 10^expon powers of _transition
     * @return limmat: limiting distribution matrix
     */
    Eigen::MatrixXf MarkovChain::limitingMat(int expon){
        Eigen::MatrixXf limmat = _transition;
        
        for(int i = 1; i<= pow(10,expon); i++){
            limmat = limmat*_transition;
        }
        return limmat;
    }

/**
 * @summary: does mat contain key?
 * @return: answer to above question, true or false for yes or no
 */
bool contains(Eigen::MatrixXf mat, float key){
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
Eigen::MatrixXf MarkovChain::numPaths(int n){
    Eigen::MatrixXf logicMat(numStates,numStates);
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
            Eigen::MatrixXf powmat = logicMat;
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
Eigen::MatrixXf MarkovChain::isReachable(void){
    Eigen::MatrixXf R(numStates,numStates);
    Eigen::MatrixXf Rcomp = Eigen::MatrixXf::Identity(numStates,numStates);
    
    Rcomp = (Rcomp + numPaths(1));
    Eigen::MatrixXf rpow = Rcomp;
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
Eigen::MatrixXf MarkovChain::communicatingClasses(void){
    Eigen::MatrixXf CC(numStates,numStates);
    Eigen::MatrixXf reachable = isReachable();
    Eigen::MatrixXf uniqueC(numStates,numStates);
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
    
    /*
     BLOCK_TO_FIX: COMPILE-TIME ERROR: no member named 'seq' in namespace 'Eigen'
    int count = 0; //how many classes?
    vector<int> deletedRows;
    for(int i = 0; i< numStates; i++ ){
            for(int j = i + 1; j < numStates; j++){
            if(CC.row(i) == CC.row(j) && not isInVec(deletedRows, j)){
                deletedRows.push_back(j);
            }
        }//end inner for
    }//end outer for
    int s = numStates - deletedRows.size();
    Eigen::Index indicesToKeep = Eigen::seq(0,s);
    int j = 0;
    for(int i = 0; i< numStates; i++){
        if( not isInVec(deletedRows, i)){
            indicesToKeep(j) = i;
            j++;
        }
    }
    uniqueC = CC(indicesToKeep, Eigen::all);
    return uniqueC;
     */
}
