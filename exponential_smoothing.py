import matplotlib.pyplot as plt

class ExponentialSmoothing:
    def __init__(self, alpha=0.3, threshold=2.0):
        """
        Initializes the Exponential Smoothing anomaly detection model.
        
        Parameters:
        - alpha (float): Smoothing factor between 0 and 1 (default is 0.3).
        - threshold (float): Threshold for detecting anomalies based on deviation from the smoothed value (default is 2.0).
        """
        # Validate input parameters
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")
        if threshold <= 0:
            raise ValueError("threshold must be a positive number.")
        
        self.alpha = alpha  # Smoothing factor
        self.threshold = threshold  # Deviation threshold for anomalies
        self.smoothed_value = None  # Initialize smoothed value

    def detect_anomaly(self, new_value):
        """
        Detects anomalies based on the new value using exponential smoothing.
        
        Parameters:
        - new_value (float): The new data point to analyze.
        
        Returns:
        - tuple: A tuple containing:
            - bool: True if the new value is an anomaly, False otherwise.
            - float: The current smoothed value.
        """
        # Check if this is the first value being processed
        if self.smoothed_value is None:
            self.smoothed_value = new_value  # Set initial smoothed value
            return False, self.smoothed_value  # No anomaly for the first value

        # Update the smoothed value using exponential smoothing
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        
        # Calculate the deviation from the smoothed value
        deviation = abs(new_value - self.smoothed_value)
        
        # Check if the deviation exceeds the threshold
        is_anomaly = deviation > self.threshold
        return is_anomaly, self.smoothed_value
    
    def plot_data(self, data_stream, anomaly_indices):
        """
        Plots the data stream and highlights the detected anomalies.
        
        Parameters:
        - data_stream (np.ndarray): The complete data stream.
        - anomaly_indices (list): Indices of detected anomalies.
        
        Returns:
        - matplotlib.figure.Figure: The plotted figure.
        """
        # Validate input data for plotting
        if not isinstance(data_stream, (list, np.ndarray)):
            raise ValueError("data_stream must be a list or numpy array.")
        if not isinstance(anomaly_indices, (list, np.ndarray)):
            raise ValueError("anomaly_indices must be a list or numpy array.")

        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Exponential Smoothing')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
        return fig

    def simulate_Exponential_Smoothing(self, data_stream):
        """
        Simulates anomaly detection on a given data stream using exponential smoothing.
        
        Parameters:
        - data_stream (np.ndarray): The data stream to analyze.
        
        Returns:
        - matplotlib.figure.Figure: The plotted figure with anomalies highlighted.
        
        Raises:
        - ValueError: If data_stream is not a list or numpy array or is empty.
        """
        # Validate input data stream
        if not isinstance(data_stream, (list, np.ndarray)):
            raise ValueError("data_stream must be a list or numpy array.")
        if len(data_stream) == 0:
            raise ValueError("data_stream cannot be empty.")

        anomaly_indices = []  # List to store indices of detected anomalies
        
        # Iterate through the data stream to detect anomalies
        for i, data_point in enumerate(data_stream):
            is_anomaly, smoothed_value = self.detect_anomaly(data_point)  # Detect anomaly for current data point
            if is_anomaly:
                anomaly_indices.append(i)  # Store the index of the anomaly

        # Plot the results
        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig

# Example usage:
if __name__ == "__main__":
    # Generate a sample data stream using the DataGenerator
    generator = DataGenerator()
    simulated_data = generator.simulate_data_stream(num_points=1000, offset=5)  # Generate data with anomalies

    exponential_smoothing_detector = ExponentialSmoothing(alpha=0.3, threshold=2.0)  # Create an instance
    exponential_smoothing_detector.simulate_Exponential_Smoothing(data_stream=simulated_data)  # Run anomaly detection