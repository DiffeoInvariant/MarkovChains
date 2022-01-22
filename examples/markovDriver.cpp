#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#include"MarkovChain.hpp"
#include "MatrixFunctions.hpp"
#include<vector>
#include"Eigen/Core"
#include<random>
#include<iostream>
using namespace std;
using namespace Markov;
/**
 *@author Emily Jakobs
 */
void dispMainMenu(){
    std::cout <<endl;
    std::cout << "Markov Solver Main Menu: Enter Choice" <<endl;
    std::cout << "1. Enter Transition Matrix/Initial Distribution" <<endl;
    std::cout << "2. Enter Transition Matrix Only" <<endl;
    std::cout << "3. Enter Initial Distribution Only" <<endl;
    std::cout << "4. Compute Stationary Distributions" <<endl;
    std::cout << "5. Compute Limiting Matrix" <<endl;
    std::cout << "6. Compute Limiting Distribution" <<endl;
    std::cout << "7. Print Matrix" <<endl;
    std::cout << "8. Generate Sequence" <<endl;
    std::cout << "9. Compute Number of Paths Between States" <<endl;
    std::cout << "10. Compute Reachable States" <<endl;
    std::cout << "11. View Communicating Classes" <<endl;
    std::cout << "12. Quit Program" << endl;
}

int main(int argc, char* argv[]){
    MarkovChain mc;
    
    bool done = false;
    while(!done){
        dispMainMenu();
        string c;
        getline(cin, c);
        int choice;
        try{
            choice = stoi(c);
        } catch(exception &msg){
            cerr << "Error: " << msg.what() << endl;
        }

    switch(choice){
        case 1:
        {
            //make matrix
            int n;
            string num;
            std::cout << "For the N X N transition matrix, what is N (how many states in the chain?)?" <<endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr <<  "Error: " <<msg.what() << endl;
                string num;
                std::cout << "For the N X N transition matrix, what is N (how many states in the chain?)?" <<endl;
                try{
                    getline(cin,num);
                    n = stoi(num);
                } catch(exception &msg){
                    cerr <<  "Error: " <<msg.what() << endl;
                }
            }
            
            
            Eigen::MatrixXd p(n,n);
            double entry;
            string e;
            std::cout << "Enter matrix elements one-by-one, by row, pressing enter each time." <<endl;
            for(int i = 0; i< n; i++){
                double rowsum = 0;
                for(int j = 0; j<n; j++){
                    
                    getline(cin, e);
                    try{
                        entry = (double)stof(e);
                        rowsum += entry;
                    } catch(exception &msg){
                        cerr << "Error: " << msg.what() << endl;
                        j-=1;
                    }
                    p(i,j) = entry;
                }//end for
                
                if(rowsum >= 1.01 || rowsum <= 0.99){
                    int resp;
                    string inresp;
                    std::cerr << "Error: Rows do not sum to 1. Press 1 to auto-normalize, 2 to continue." << endl;
                    getline(cin, inresp);
                    try{
                        resp = stoi(inresp);
                    } catch(exception &msg){
                        cerr << msg.what() <<endl;
                        std::cout << "Enter 1 or 2 only, followed by enter." <<endl;
                    }
                    switch(resp){
                        case 1:
                        {
                            for(int j = 0; j<n; j++){
                                p(i,j) /= rowsum;
                            }
                            break;
                        }
                        case 2:
                        {
                            break;
                        }
                    }//end switch
                }//end if
                
            }//end for
            
            //make initial distribution
            Eigen::MatrixXd init(1,n);
            std::cout << "Now, enter the initial distribution. Enter " << n << " numbers,  pressing enter each time" << endl;
            for(int i = 0; i<n; i++){
                getline(cin, e);
                try{
                    entry = (double)stof(e);
                    init(0,i) = entry;
                } catch(exception &msg){
                    cerr <<  "Error: " << msg.what() << endl;
                    i-=1;
                }
                
            }
            init = init/(init.sum());
            mc.setModel(p,init,n);
            break;
        }//end case 1
            
        case 2:
        {
            
            //make matrix
            int n;
            string num;
            std::cout << "For the N X N transition matrix, what is N?" <<endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr <<  "Error: " <<msg.what() << endl;
                string num;
                std::cout << "For the N X N transition matrix, what is N?" <<endl;
                try{
                    getline(cin,num);
                    n = stoi(num);
                } catch(exception &msg){
                    cerr <<  "Error: " <<msg.what() << endl;
                    break;
                }
            }
            
            Eigen::MatrixXd p(n,n);
            double entry;
            string e;
            std::cout << "Enter matrix elements one-by-one, by row, pressing enter each time." <<endl;
            for(int i = 0; i< n; i++){
                double rowsum = 0;
                for(int j = 0; j<n; j++){
                    
                    getline(cin, e);
                    try{
                        entry = (double)stof(e);
                        rowsum += entry;
                    } catch(exception &msg){
                        cerr << "Error: " << msg.what() << endl;
                        j-=1;
                    }
                    p(i,j) = entry;
                }//end for
                //check that rows sum to 1
                if(rowsum >= 1.01 || rowsum <= 0.99){
                    int resp;
                    string inresp;
                    std::cout<< "Error: Row does not sum to 1. Press 1 to auto-normalize, 2 to continue.";
                    getline(cin, inresp);
                    try{
                        resp = stoi(inresp);
                    } catch(exception &msg){
                        cerr << msg.what() <<endl;
                        std::cout << "Enter 1 or 2 only, followed by enter." <<endl;
                    }
                    switch(resp){
                        case 1:
                        {
                            for(int j = 0; j<n; j++){
                                p(i,j) /= rowsum;
                            }
                            break;
                        }
                        case 2:
                        {
                            break;
                        }
                    }//end switch
                }//end if
                
            }//end for
            mc.setTransition(p);
            mc.setNumStates(n);
            break;
        }
        case 3:
        {
            int n;
            string num;
            std::cout << "How many states are there?" <<endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr <<  "Error: " << msg.what() << endl;
            }
            //make initial distribution
            string e;
            double entry;
            Eigen::MatrixXd init(1,n);
            std::cout << "Now, enter the initial distribution. Enter " << n << " numbers,  pressing enter after each." << endl;
            for(int i = 0; i<n; i++){
                getline(cin, e);
                try{
                    entry = (double)stof(e);
                    init(0,i) = entry;
                } catch(exception &msg){
                    cerr << "Error: " << msg.what() << endl;
                    i-=1;
                }
                
            }
            try{
                init = init/(init.sum());
            }catch(exception &msg){
                cerr << "Error: " << msg.what() <<endl;
            }
            mc.setInitial(init);
            break;
        }
        
        case 4:
        {
            mc.stationaryDistributions();
            break;
        }
        case 5:
        {
            int n;
            string num;
            std::cout << "Calculate to how many powers of the matrix?" << endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr <<  "Error: " << msg.what() << endl;
            }
            std::cout << "The limiting matrix is " <<endl << mc.limitingMat(n) <<endl;
            break;
        }
        case 6:
        {
            int n;
            string num;
            std::cout << "Calculate to how many powers of the matrix?" << endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr <<  "Error: " << msg.what() << endl;
            }
            std::cout << "The limiting distribution is ";
            std::cout << mc.limitingDistribution(n);
            break;
        }
        case 7:
        {
            cout << mc <<endl;
            break;
        }
        case 8:
        {
            int n;
            string num;
            if(mc.getInit().cols() <= 1){
                std::cout << "Error: Enter an initial distribution." << endl;
                break;
            }
            std::cout << "How many elements do you want in the sequence?" <<endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr << "Error: " << msg.what() << endl;
            }
            
            std::cout << "Generated sequence of length " << n <<":" << endl;
            vector<int> res(n);
            try{
                res = mc.generateSequence(n);
                for(int x = 0; x < res.size(); x++){
                    std::cout << res[x] << endl;
                }
            } catch(exception &msg){
                cerr << "Error: " << msg.what() << endl;
                cout << "You may have not set an initial distribution. Ensure one is set." << endl;
            }
            
            break;

        }
        case 9:
        {
            int n;
            string num;
            std::cout << "We will compute the matrix where element (i,j) is the number of permissible paths of length " << endl;
            std::cout << "N from state i to state j. What is N?" << endl;
            getline(cin,num);
            try{
                n = stoi(num);
            } catch(exception &msg){
                cerr << "Error: " << msg.what() << endl;
            }
            cout << mc.numPaths(n) << endl;
            break;
        }
        case 10:
        {
            
            std::cout << "We will compute the matrix where element (i,j) is 1 if state j can be reached from state i, 0 else." << endl;
            
            std::cout << mc.isReachable() << endl;
        }
        case 11:
        {
            Eigen::MatrixXd cc = mc.communicatingClasses();
            cout << cc <<endl;
            break;
        }
        case 12:
        {
            done = true;
            break;
        }
    }//end switch
    }//end while
    
}//end main

