
#include "MarkovFunctions.hpp"
#include<iostream>
#include<utility>
using namespace Markov;
using namespace std;
//functorized increment; increments argument by step
template<typename T>
class increment
{
private:
    T step;
public:
    increment(T n) : step(n) {}
    T operator()(T in_num) const
    {
        return in_num + step;
    }
};
int main(){
    //test pipe closure, add one
    const pipe_closure<add_one_f> add_one = {};
    int num3 = 3;
    std::cout << "Testing add_one_f. Should return 4:" << endl;
    int num4 = num3 | add_one;
    std::cout << num4 << endl;
    
    Eigen::MatrixXd maxmat(2,2);
    maxmat << 0,4,
              2,8;
    
    const pipe_closure<real_matrix_max_f> matMax = {};
    std::cout << "Testing real_matrix_max_f. Should return 8:" << endl;
    auto max = maxmat | matMax;
    std::cout << max << endl;
    
    std::cout << "Testing real_matrix_max_f | add_one. Should return 9:" << endl;
    auto maxp1 = maxmat | matMax | add_one;
    std::cout << maxp1 << endl;
    
    std::cout<< "Testing mat_apply_f with increment on maxmat." << endl;
    const pipe_closure<mat_apply_f> mapply = {};
    increment<double> incrFun(1.5);
    auto incrementedMat = make_mapply_pair<Eigen::MatrixXd, increment<double> >(maxmat, incrFun) | mapply;
    std::cout << incrementedMat << endl;
    
    std::cout << "Testing pipable on add_one. Should return 6:" << endl;
    const constexpr pipable<add_one_f> AOT = {};
    auto val = num4 | AOT | AOT;
    cout << val << endl;
    
    return 0;
}
