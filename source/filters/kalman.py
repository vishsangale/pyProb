"""
Class implementing basic single state Kalman filter.
"""


class KalmanFilter(object):
    """
        Kalman filter implementation.
    """

    def __init__(self, A, B, C, x, P, Q, R):
        self.process_dynamics = A
        self.control_dynamics = B
        self.measurement = C
        self.current_state_estimate = x
        self.current_probability_estimate = P
        self.process_covariance = Q
        self.measurement_covariance = R

    def kalman_step(self, control, measurement):
        """
        Kalman step for each iteration, includes state prediction, observation prediction, and
        observation state update.
        Args:
            control: control input to the current state
            measurement: measurement input from the current state

        Returns:

        """
        predicted_state_estimate, predicted_probability_estimate = self._prediction_step(control)

        innovation, innovation_covariance = self._observation_step(measurement,
                                                                   predicted_state_estimate,
                                                                   predicted_probability_estimate)

        self._update_step(predicted_probability_estimate, predicted_state_estimate, innovation,
                          innovation_covariance)

    def _prediction_step(self, control):
        # A*x + B*u
        predicted_state_estimate = self.process_dynamics * self.current_state_estimate + \
                                   self.control_dynamics * control
        # (A*x)*A + Q
        predicted_probability_estimate = self.process_dynamics * \
                                         self.current_probability_estimate * \
                                         self.process_dynamics + self.process_covariance
        return predicted_state_estimate, predicted_probability_estimate

    def _observation_step(self, measurement, predicted_state_estimate,
                          predicted_probability_estimate):
        innovation = measurement - self.measurement * predicted_state_estimate
        # C*
        innovation_covariance = self.measurement * predicted_probability_estimate * \
                                self.measurement + self.measurement_covariance

        return innovation, innovation_covariance

    def _update_step(self, predicted_probability_estimate, predicted_state_estimate, innovation,
                     innovation_covariance):
        kalman_gain = (predicted_probability_estimate * self.measurement) / innovation_covariance
        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
        self.current_probability_estimate = (1 - kalman_gain * self.measurement) * \
                                            predicted_probability_estimate
