
#ifndef CFTP_hpp
#define CFTP_hpp

#include "CFTP.hpp"
#include "Eigen/Core"
/**
 * Coupling From The Past to perfectly sample from a Markov chain matrix mat
 */

Eigen::MatrixXd matPow(Eigen::MatrixXd &mat, int _pow);

//void *threadedMatPow(Eigen::MatrixXd &mat, int pow);

double variation_distance(Eigen::MatrixXd dist1, Eigen::MatrixXd dist2);

double k_self_variation_distance(Eigen::MatrixXd trans, int k);
int mixing_time(Eigen::MatrixXd &trans);

int isCoalesced(Eigen::MatrixXd &mat);

int random_transition(Eigen::MatrixXd &mat, int init_state, double r);

/**
 * @author: Zane Jakobs
 * @summary: voter CFTP algorithm to perfectly sample from the Markov chain with transition matrix mat. Algorithm from https://pdfs.semanticscholar.org/ef02/fd2d2b4d0a914eba5e4270be4161bcae8f81.pdf
 * @return: perfect sample from matrix's distribution
 */
int voter_CFTP(Eigen::MatrixXd &mat);

#endif /* CFTP_hpp */
