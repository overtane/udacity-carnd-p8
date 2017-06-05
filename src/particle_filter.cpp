/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *  Author: Tiffany Huang
 *
 *  Particle filter implementation: Jul 5, 2017
 *  Author: Olli Vertanen
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static std::default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

    // Set the number of particles
    num_particles = 50;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i=0; i < num_particles; i++) {
        Particle p;

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	// Add measurements to each particle and add random Gaussian noise.

    std::default_random_engine gen;

    for (int i=0; i < num_particles; i++) {
        double theta = particles[i].theta;
        double yaw_angle = yaw_rate*delta_t;

        // add measurements
        if (fabs(yaw_rate) < 0.001) {
            // motion model: going straight
            particles[i].x += velocity * delta_t * cos(theta);
            particles[i].y += velocity * delta_t * sin(theta);
        } else {
            particles[i].x += velocity/yaw_rate * (sin(theta+yaw_angle)-sin(theta));
            particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta+yaw_angle));
            particles[i].theta += yaw_angle;
        }

        // add random Gaussian noise
        normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

        particles[i].x     = dist_x(gen);
        particles[i].y     = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
    // Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.

    for (int i=0; i < observations.size(); i++) {

        int closest_id = -1;
        double closest_dist = std::numeric_limits<double>::max();

        for (int j=0; j < predicted.size(); j++) {
            double d = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if (d < closest_dist) {
                closest_id = predicted[j].id;
                closest_dist = d;
            }
        }

        observations[i].id = closest_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    //
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement (look at equation 
	// 3.33,  http://planning.cs.uiuc.edu/node99.html)

    double tot_weights = 0.0;

    weights.clear();

    // 'constant' terms for probability calculations
    double sigma_x_squared = 2 * std_landmark[0] * std_landmark[0];
    double sigma_y_squared = 2 * std_landmark[1] * std_landmark[1];
    double dvdr = 2 * M_PI * std_landmark[0] * std_landmark[1];


    for (int i=0; i < num_particles; i++) {
        
        // Predict measurements to all map landmarks within sensor range
        // For each predicted particle, we search for map landmarks that are within the sensor range
        std::vector<LandmarkObs> predicted;

        for (int j=0; j < map_landmarks.landmark_list.size(); j++) {
            double d = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
            if (d < sensor_range) {
                LandmarkObs l;
                l.x = map_landmarks.landmark_list[j].x_f;
                l.y = map_landmarks.landmark_list[j].y_f;
                l.id = map_landmarks.landmark_list[j].id_i;
                predicted.push_back(l);
            }
        }

        // Transform observations to map coordinate space
        // Note: resulting observations are from predicted particle's point of view

        vector<LandmarkObs> observations_tr; // transformed observations
        for (int j=0; j < observations.size(); j++) {
            LandmarkObs tr;
            tr.x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
            tr.y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
            tr.id = 0;
            observations_tr.push_back(tr);
        } 

        // Associate the sensors measurements to predicted map landmarks
        dataAssociation(predicted, observations_tr);
      
        // Calculate multi-variate Gaussian probability for each observation
        particles[i].weight = 1.0;

        for (int j=0; j < observations_tr.size(); j++) {
             
            double pre_x, pre_y; // predicted location
            double obs_x, obs_y; // measured location
            
            obs_x = observations_tr[j].x;
            obs_y = observations_tr[j].y;

            // find prediction associated to the measurement
            for (int k=0; k < predicted.size(); k++) {
                if (observations_tr[j].id == predicted[k].id) {
                    pre_x = predicted[k].x;
                    pre_y = predicted[k].y;
                    break;
                }
            }

            // Probability of this observation
            double p = exp(-((obs_x-pre_x)*(obs_x-pre_x)/sigma_x_squared + (obs_y-pre_y)*(obs_y-pre_y)/sigma_y_squared)) / dvdr;

            // Particle's final weight is the product of calculated probabilities to all observations
            particles[i].weight *= p;
        }

        tot_weights += particles[i].weight;
        // collect weights for resampling
        weights.push_back(particles[i].weight);
    }

    // Normalize weights so that sum of all weights is one
    for (int i=0; i < num_particles; i++) {
        particles[i].weight /= tot_weights; 
    }
}

void ParticleFilter::resample() {

	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::discrete_distribution<int> distr(weights.begin(), weights.end());

    vector<Particle> resampled;
    for(int i=0; i < num_particles; i++) {
        int n = distr(gen);
        resampled.push_back(particles[n]);
    }

    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
