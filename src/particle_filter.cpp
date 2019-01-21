#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "multiv_gauss.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   *   Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   *   Add random Gaussian noise to each particle.
   *
   *   NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */

  num_particles = 1000;  // Set the number of particles
  std::default_random_engine gen;
  double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

  // Set standard deviations for x, y, and theta
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std_x);

  // Normal (Gaussian) distributions for y
  std::normal_distribution<double> dist_y(y, std_y);

  // Normal (Gaussian) distributions for theta
  std::normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p = {};
    p.id = i + 1;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1;

    weights.push_back(1);

    particles.push_back(p);
  }
  is_initialized = true;
  cout<<"Finished init" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   *  Add measurements to each particle and add random Gaussian noise.
   *
   *  NOTE: When adding noise you may find std::normal_distribution
   *  and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::default_random_engine gen;
   double std_x, std_y, std_theta;  // Standard deviations for x, y, and std_theta
   std_x = std_pos[0];
   std_y = std_pos[1];
   std_theta = std_pos[2];
   for (int i =0; i < num_particles; i++) {
     double x0, y0, theta0;
     double xf, yf, thetaf;

     x0 = particles[i].x;
     y0 = particles[i].y;
     theta0 = particles[i].theta;

     xf = x0 + (velocity / yaw_rate) * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
     yf = y0 + (velocity / yaw_rate) * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
     thetaf = theta0 + yaw_rate * delta_t;

     // Normal (Gaussian) distribution for xf
     std::normal_distribution<double> dist_x(xf, std_x);

     // Normal (Gaussian) distributions for yf
     std::normal_distribution<double> dist_y(yf, std_y);

     // Normal (Gaussian) distributions for thetaf
     std::normal_distribution<double> dist_theta(thetaf, std_theta);

     particles[i].x = dist_x(gen) + xf;
     particles[i].y = dist_y(gen) + yf;
     particles[i].theta = dist_theta(gen) + thetaf;
   }
   cout << "Finished prediction" << endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   *
   *   NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
   for (int i = 0; i < observations.size(); i++) {
     int closest_landmark = -1;
     int min_dist = 99999999;
     int curr_dist;
     for (int j = 0; j < predicted.size(); j++) {
       curr_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
       if (curr_dist < min_dist) {
         min_dist = curr_dist;
         closest_landmark = predicted[j].id;
       }
     }
     observations[i].id = closest_landmark;
   }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a multi-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   *
   *   NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   double normalizer = 0;

   // Iterate for each particle
   for (int i = 0; i < num_particles; i++) {
     Particle p = particles[i];
     int num_obs = observations.size();

     // Transform the observations from vehicle coordinates to
     // map coordinates.
     vector<LandmarkObs> transformed_observations;
     for (int j = 0; j < observations.size(); j++) {
       LandmarkObs transformed_obs = transform_obs(p.x, p.y, p.theta, observations[j]);
       transformed_observations.push_back(transformed_obs);
     }

     // Create a vector of LandmarkObs of landmarks from the map
     vector<LandmarkObs> landmarks;
     for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double dist_to_landmark = dist(
        map_landmarks.landmark_list[j].x_f,
        map_landmarks.landmark_list[j].y_f,
        p.x,
        p.y
      );
      if (dist_to_landmark <= sensor_range) {
        LandmarkObs landmark = {
          map_landmarks.landmark_list[j].id_i,
          map_landmarks.landmark_list[j].x_f,
          map_landmarks.landmark_list[j].y_f,
        };
        landmarks.push_back(landmark);
      }
     }

     // Associated the transformed map observations for this particle
     // with the nearest landmark.
     dataAssociation(landmarks, transformed_observations);

     // Save the landmark id, landmark x, landmark y as the particle's
     // properties.
     vector<int> associations;
     vector<double> sense_x;
     vector<double> sense_y;
     for (int j = 0; j < num_obs; j++) {
       LandmarkObs landmark;
       for (int k = 0; k < landmarks.size(); k++ ) {
         if (landmarks[k].id == transformed_observations[j].id) {
           landmark = landmarks[k];
           break;
         }
       }
       associations.push_back(landmark.id);
       sense_x.push_back(landmark.x);
       sense_y.push_back(landmark.y);
     }
     SetAssociations(particles[i], associations, sense_x, sense_y);

     // Calculate the new weight for this particle
     double weight = 1;
     for (int j = 0; j < num_obs; j++) {
       double sig_x = std_landmark[0];
       double sig_y = std_landmark[1];
       double x_obs = transformed_observations[j].x;
       double y_obs = transformed_observations[j].y;
       double mu_x = sense_x[j];
       double mu_y = sense_y[j];
       double pxy = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
       weight *= pxy;
     }

     particles[i].weight = weight;
     weights[i] = weight;

     normalizer += weight;
   }
   for (int i =0; i < num_particles; i++) {
     particles[i].weight /= normalizer;
     weights[i] /= normalizer;
   }

   cout <<"Finished updateWeights"<<endl;
}

void ParticleFilter::resample() {
  /**
   *   Resample particles with replacement with probability proportional
   *   to their weight.
   *
   *   NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   double w_max = *max_element(weights.begin(), weights.end());
   cout << "w_max: " << w_max << endl;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0, 2 * w_max);

   vector<Particle> new_particles;
   vector<double> new_weights;
   int w_index = rand() % num_particles;

   for (int i = 0; i < num_particles; i ++) {
     double weight = weights[w_index];
     double beta = distribution(generator);

     while (weight < beta) {
       beta -= weight;
       w_index = (w_index + 1) % num_particles;
       weight = weights[w_index];
     }
     new_particles.push_back(particles[w_index]);
     new_weights.push_back(particles[w_index].weight);
   }

   particles = new_particles;
   weights = new_weights;
   cout << "Finished resample. There are "<<new_particles.size() << " new particles" << endl;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
