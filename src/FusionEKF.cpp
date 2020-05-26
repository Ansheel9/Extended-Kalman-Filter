#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#define EPS 0.0001 

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */

FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;
  // Initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
			  0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);
  
  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /**
   * Initialization
   */
  if (!is_initialized_) {
    ekf_.x_ = VectorXd(4);
    // first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0]; // range
	  float phi = measurement_pack.raw_measurements_[1]; // bearing
	  float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
	  // Coordinates convertion from polar to cartesian
	  float x = rho * cos(phi); 
	  float y = rho * sin(phi);
	  float vx = rho_dot * cos(phi);
	  float vy = rho_dot * sin(phi);
	  ekf_.x_ << x, y, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // We don't know velocities from the first measurement of the LIDAR, so, we use zeros
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    
    // Deal with the special case initialisation problems
    if (fabs(ekf_.x_(0)) < EPS and fabs(ekf_.x_(1)) < EPS){
		ekf_.x_(0) = EPS;
		ekf_.x_(1) = EPS;
	}
    
	// Initial covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
			   0, 1, 0, 0,
			   0, 0, 1000, 0,
			   0, 0, 0, 1000;
    cout << "EKF init: " << ekf_.x_ << endl;
    // Save the initiall timestamp for dt calculation
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  
  // Calculate the timestep between measurements in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt /= 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // State transition matrix update
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
			0, 1, 0, dt,
			0, 0, 1, 0,
			0, 0, 0, 1;	
  // Noise values from the task
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt; 
  float dt_4 = dt_3 * dt; 
  float dt_4_4 = dt_4 / 4; 
  float dt_3_2 = dt_3 / 2; 
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
	         0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
	         dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
 	         0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else {
    // Laser updates
	ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
	ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}