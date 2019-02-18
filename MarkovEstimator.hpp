//
//  MarkovEstimator.hpp
//  
//
//  Created by Zane Jakobs on 2/13/19.
//

#ifndef MarkovEstimator_hpp
#define MarkovEstimator_hpp

#include "MarkovChain.hpp"
#include "Eigen/Core"
#include "CFTP.cpp"
#include<cmath>
#include<iostream>
#include<vector>
#include<random>
#include<algorithm>
#include<list>
#include<limits>
#include<deque>
using namespace std;
using namespace Eigen;
/**
 * @author: Zane Jakobs
 * @summary: stores a sequence of states
 */

typedef struct
{
    std::vector<int> seq;
    void set_seq(std::vector<int> _seq){
        seq = _seq;
    }
}Sequence;
/**
 * @author: Zane Jakobs
 * @summary: stores a vector of sequences of states
 */
typedef struct
{
    std::list<Sequence> dat;
    void set_dat(std::list<Sequence> _dat){
        dat = _dat;
    }
    
}DataBase;

/**
 * @author: Zane Jakobs
 * @summary: stores results of regression
 */
typedef struct
{
    bool converged = false; //did we converge to a (local) optimum?
    Eigen::MatrixXd transition; //transition matrix
    /*
     A matrix, where mat(i,j) = 1 if j is in the i-th communicating class, 0 else
     */
    Eigen::MatrixXd communicating_classes;
} MarkovEstimatorResults;

typedef struct
{
    int seqLen;//length of sequences
    int oversize = 100; // upper bound on number of states
    int numParams; //number of parameters after identifying zeros
    int numBatches; //how many batches?
    double learningRate = 1.0e-3; //max size of an adjustment
    double tol = 1.0e-7; //convergence tolerance
    int max_iter = 1.0e+5; //max iterations
    bool isbatches = true; //number of batches, or max size?
    Eigen::MatrixXd initialParams;
}ModelParams;
class MarkovEstimator{
    
public:
    
    //default constructor
    MarkovEstimator(DataBase _data){
        //make default res
        MarkovEstimatorResults defRes;
        defRes.converged = false;
        defRes.transition = MatrixXd::Zero(1,1);
        defRes.communicating_classes = MatrixXd::Zero(1,1);
        
        //make default modPars
        
        ModelParams defMP;
        defMP.seqLen = 1;//length of sequences
        defMP.oversize = 100; // upper bound on number of states
        defMP.numParams = 1; //number of parameters after identifying zeros
        defMP.learningRate = 1.0e-3; //max size of an adjustment
        defMP.tol = 1.0e-7; //convergence tolerance
        defMP.max_iter = 1.0e+5; //max iterations
        defMP.initialParams = MatrixXd::Zero(1,1); //initial parameters
        this->data = _data;
        this->res = defRes;
        this->modPars = defMP;
    }
    
    //fully parametrized constructor
    /*MarkovEstimator::MarkovEstimator(DataBase _data,MarkovEstimatorResults _res, ModelParams _modPars){
     this.data = _data;
     this.res = _res;
     this.modPars = _modPars;
     }*/
    
    /**
     * @author: Zane Jakobs
     * @summary: makes all sequences in a database the same length by dropping all that do not have length n
     * @param n: length of sequences
     * @param df: database to clean
     
     std::vector<DataBase> MarkovEstimator::cleanData(std::vector<DataBase> df, int n){
     int count = 0;
     for(iterator it = df.begin(); it != df.end(); it++){
     if(*it.dat.size() != n){
     .dat.erase(it);
     count++;
     }
     }
     if(count > 0){
     cout << "Deleted " << count << " sequences of the wrong length." <<endl;
     }
     return df;
     }
     */
    static Eigen::MatrixXd normalize(Eigen::MatrixXd &mat){
        for(int i = 0; i < mat.rows(); i++){
            mat.row(i) = mat.row(i)/(mat.row(i).sum());
        }
        return mat;
    }
    /**
     * @author: Zane Jakobs
     * @summary: computes transition matrix based on empirical distribution of values. In
     * particular, attempts to find which entries should be zero
     * @param dat: the data
     * @param oversize: upper bound on the number of states in the chain
     * @return: estimate of transition matrix, with zero entries where they should be to
     * preserve communicating classes
     */
    static Eigen::MatrixXd distMatrix(DataBase df, int oversize = 100){
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(oversize,oversize);
        
        //loop through data, entering values
        int largest = 0; //number of unique states
        for(auto it = df.dat.begin(); it != df.dat.end(); it++){
            for(auto jt = (*it).seq.begin(); jt++ != (*it).seq.end(); jt++){
                auto pos = jt;
                int s1 = *pos;
                int s2 = *(pos++);
                mat(s1,s2) = mat(s1,s2) +1;
                if(s1 > largest || s2 > largest){
                    s1 > s2 ? largest = s1 : largest = s2;
                }
            }
        }
        largest++;
        /*
         set matrix size to largest x largest, preserving values
         */
        mat.conservativeResize(largest,largest);
        mat = normalize(mat);
        return mat;
    }
    /**
     * @author: Zane Jakobs
     * @summary: splits database into either n batches, or batches of size up to n
     * sequences each
     * @param n: number of bins, or max size of bin
     * @param isbins: if true, n is number of bins. Else, n is max size of bin
     * @return: vector of databases
     */
    std::vector<DataBase> splitData(int n, bool isbins = true){
        std::vector<DataBase> split(n);
        if(isbins){
            
            if(n < data.dat.size()){
                throw "Error: cannot separate data into that many batches. Choose n <= size of data";
                return split;
            }
            
            std::vector<DataBase> split(n);
            std::vector<int> chosen(n); //sequences we've already chosen
            //set random seed
            random_device rd;
            //init Mersenne Twistor
            mt19937 gen(rd());
            // unif(0,1)
            uniform_real_distribution<> dis(0.0,1.0);
            
            for(int i = 0; i< n; i++){
                int index = 1;
                double rvalue = dis(gen);
                //choose int values from 1 to data.dat.size() uniformly
                while(rvalue * index < data.dat.size()){
                    index++;
                }
                auto f1 = std::find(chosen.begin(), chosen.end(), index);
                if(f1 != chosen.end()){
                    i--;
                }
                else{
                    int i = 0;
                    auto it = data.dat.begin();
                    for(i = 0; i < index; i++){
                        it++;
                    }
                    split[i].dat.push_back(*it);
                    chosen.push_back(index);
                }
            }
        }//end if
        else{
            int nbins = int( double(data.dat.size())/double(n) +1);
            std::vector<DataBase> split(nbins);
            std::vector<int> chosen(nbins); //sequences we've already chosen
            //set random seed
            random_device rd;
            //init Mersenne Twistor
            mt19937 gen(rd());
            // unif(0,1)
            uniform_real_distribution<> dis(0.0,1.0);
            
            for(int i = 0; i< nbins; i++){
                int index = 1;
                double rvalue = dis(gen);
                //choose int values from 1 to data.dat.size() uniformly
                while(rvalue * index < data.dat.size()){
                    index++;
                }
                auto f2 = std::find(chosen.begin(), chosen.end(), index);
                if(f2 != chosen.end()){
                    i--;
                }
                else{
                    int i = 0;
                    auto it = data.dat.begin();
                    for(i = 0; i < index; i++){
                        it++;
                    }
                    split[i].dat.push_back(*it);
                    chosen.push_back(index);
                }
            }//end for
            
        }//end else
        return split;
    }
    
    
    
    
    /**
     * @author: Zane Jakobs
     * @summary: computes empirical distribution of data
     * @param dat: data
     * @param oversize: upper bound on the number of states in the chain
     * @return: estimate of limiting distribution of the Markov chain
     */
    Eigen::MatrixXd empiricalDist(DataBase df, int oversize = 100){
        
        Eigen::MatrixXd distr = Eigen::MatrixXd::Zero(1,oversize);
        
        //loop through data, entering values
        int largest = 0; //number of unique states
        int s1= 0;
        int s2 = 0;
        for(auto it = df.dat.begin(); it != df.dat.end(); it++){
            s2 = 0;
            for(auto jt = (*it).seq.begin(); jt++ != (*it).seq.end(); jt++){
                distr(0,*jt) = distr(0,*jt) +1;
                if(s1 > largest || s2 > largest){
                    s1 > s2 ? largest = s1 : largest = s2;
                }
                s2++;
            }
            s1++;
        }
        largest++;
        distr.conservativeResize(1,largest); //resize, keeping data
        distr = distr/distr.sum(); //normalize
        return distr;
    }
    
    //computes limit matrix
    Eigen::MatrixXd limitingMatrix(int expon, Eigen::MatrixXd &mat){
        Eigen::MatrixXd limmat;
        limmat = mat.array().pow(expon).matrix();
        return limmat;
    }
    /**
     * @author: Zane Jakobs
     * @summary: error function, defined as sum of (pi_i - colavg_i), with
     * pi the empirical distribution, colsum the average of each column.
     * @return:sum of squares of (pi_i - colsavg_i) with pi the empirical distribution, colavg the average of each column
     */
    double errorFunc(MatrixXd &trans, MatrixXd &dist){
        if(dist.cols() != trans.cols()){
            throw "Error: Distribution and transition matrix have different number of states";
            return std::numeric_limits<double>::max();
        }
        int states = trans.cols();
        Eigen::MatrixXd colavg(1,states);
        static const int matpow = 10;
        Eigen::MatrixXd powmat = limitingMatrix(matpow, trans);
        for(int i = 0; i< states; i++){
            colavg(0,i) = (powmat.col(i)).sum()/(double(states));
        }
        
        if(dist.cols() > 1){
            Eigen::MatrixXd x = dist - colavg;
            return (x * x.transpose()).sum();
        }
        else{
            Eigen::MatrixXd x = dist.transpose() - colavg;
            return (x * x.transpose()).sum();
        }
    }
    
    /**
     * @author: Zane Jakobs
     * @return: matrix of partial derivatives of error function wrt each nonzero parameter
     */
    Eigen::MatrixXd gradient(MatrixXd &trans, MatrixXd &dist){
        int states  = trans.cols();
        Eigen::MatrixXd grad(states,states);
        if(dist.cols() != trans.cols()){
            throw "Error: Distribution and transition matrix have different number of states";
            return grad;
        }
        static const double h = 1.0e-5; //as in the difference quotient, ((f(x+h)-f(x))/h
        
        Eigen::MatrixXd transCpy = trans;
        for(int i = 0; i < states; i++){
            for(int j = 0; j< states; j++){
                if(trans(i,j) != 0){
                    transCpy(i,j) = transCpy(i,j) + h;
                    grad(i,j) = (errorFunc(transCpy, dist) - errorFunc(trans,dist))/(h); //difference quotient
                    transCpy(i,j) = transCpy(i,j) - h;
                }
                else{
                    grad(i,j) = 0;
                }
            }//end inner for
        }//end outer for
        
        return grad;
    }
    
    /**
     * @author: Zane Jakobs
     * @summary: sets parameters of model. See struct ModelParams for more info on parameters
     * @param nbatches: number of bins, or max size of bin
     * @param isbatches: if true, nbatches is number of batches. Else, nbatches is max size of batches
     * @param oversize: upper bound on number of states in the chain
     * @param lRate: learning rate. If not set, model decides learning rate
     * @param sl: sequence length
     * @return: nothing. Sets model params.
     */
    void setModel( int nbatches, bool _isbatches = true, int _oversize = 100, double lRate = 0, int _max_iter = 1.0e+5, double _tol = 1.0e-7, int sl = 0){
        //split data
        (sl == 0) ? modPars.seqLen = (*data.dat.begin()).seq.size() : modPars.seqLen = sl;//length of sequences
        
        modPars.oversize = _oversize; // upper bound on number of states
        
        Eigen::MatrixXd initMat = distMatrix(data,sl); //
        
        int count = 0;
        for(int i = 0; i< initMat.cols(); i++){
            for(int j = 0; j < initMat.rows(); j++){
                if(initMat(i,j) == 0){
                    count++;
                }
            }
        }
        int imcols = initMat.cols();
        modPars.numParams = imcols * imcols - count; //number of parameters after identifying zeros
        (lRate == 0) ? modPars.learningRate = 1.0e-3 : modPars.learningRate = lRate; //max size of an adjustment
        
        
        (sl == 0) ? modPars.seqLen = (*data.dat.begin()).seq.size() : modPars.seqLen = sl;//length of sequences
        modPars.tol = _tol; //convergence tolerance
        modPars.max_iter = _max_iter; //max iterations
        modPars.initialParams = initMat; //initial parameters
        
        modPars.isbatches = _isbatches;
        modPars.numBatches = nbatches;
        
    }
    
    /**
     * @author: Zane Jakobs
     * @summary: trains the model
     */
    void train(){
        //split data into batches
        std::vector<DataBase> batch_data =  splitData(modPars.numBatches, modPars.isbatches);
        //clean split data
        //batch_data = cleanData(batch_data, modPars.seqLen);
        //get initial matrix
        Eigen::MatrixXd predictor = distMatrix(data, modPars.oversize);
        
        Eigen::MatrixXd batch_dist;
        //train on each batch
        bool hit_tolerance = false; //true if prev_step_size gets within the tolerance
        for(int i = 0; i < batch_data.size(); i++){
            //get distribution of batches
            batch_dist = empiricalDist(batch_data[i], modPars.oversize);
            //vars to control when gradient descent stops
            bool exploding_gradient = false; //true if gradient is infinite
            
            int iterations = 0; //how many times have we iterated?
            double prev_step_size = 1.0;
            Eigen::MatrixXd grad(modPars.seqLen, modPars.seqLen);//gradient matrix
            
            //gradient descent on the batch
            while(iterations < modPars.max_iter && prev_step_size > modPars.tol && not exploding_gradient && not hit_tolerance){
                grad = gradient(predictor, batch_dist);
                predictor = predictor - modPars.learningRate * grad;
                prev_step_size = modPars.learningRate * grad.maxCoeff();
                normalize(predictor);
            }//end while
            
        }//end for
        res.converged = hit_tolerance;
        res.transition = predictor;
    }

private:
    DataBase data;
    MarkovEstimatorResults res;
    ModelParams modPars;

    
    
    
    
};


#endif /* MarkovEstimator_hpp */
