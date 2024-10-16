import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from collections import deque

class Z_Score:
    def __init__(self) -> None:
        """
        Initializes the Z_Score anomaly detection class.
        """
        pass

    def detect_anomalies(self, data, threshold=3):
        """
        Detects anomalies in the provided data using Z-Score method.

        Parameters:
        - data (np.ndarray): Input data array to analyze.
        - threshold (float): Z-Score threshold for determining anomalies (default is 3).

        Returns:
        - np.ndarray: Indices of detected anomalies.

        Raises:
        - ValueError: If data is not a numpy array or is empty.
        """
        # Validate input data
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input data cannot be empty.")

        # Calculate Z-Scores
        z_scores = zscore(data)
        
        # Identify anomalies based on Z-Score threshold
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        return anomalies

    def plot_data(self, data_stream, anomaly_indices):
        """
        Plots the data stream and highlights the detected anomalies.

        Parameters:
        - data_stream (np.ndarray): The complete data stream.
        - anomaly_indices (np.ndarray): Indices of detected anomalies.

        Returns:
        - matplotlib.figure.Figure: The plotted figure.
        """
        # Validate input data for plotting
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if not isinstance(anomaly_indices, (list, np.ndarray)):
            raise ValueError("anomaly_indices must be a list or numpy array.")

        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Moving Z-Score')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
        return fig

    def simulate_Z_score_detection(self, data_stream, window_size, threshold):
        """
        Simulates anomaly detection on a continuous data stream using a sliding window approach.

        Parameters:
        - data_stream (np.ndarray): The data stream to analyze.
        - window_size (int): Size of the sliding window for Z-Score calculation.
        - threshold (float): Z-Score threshold for detecting anomalies.

        Returns:
        - matplotlib.figure.Figure: The plotted figure with anomalies highlighted.

        Raises:
        - ValueError: If data_stream is not a numpy array or if window_size is less than 1.
        """
        # Validate input data
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if data_stream.size == 0:
            raise ValueError("data_stream cannot be empty.")

        data_buffer = deque(maxlen=window_size)  # Create a fixed-size buffer
        anomaly_indices = []  # List to store indices of detected anomalies

        # Iterate over the data stream
        for i, data_point in enumerate(data_stream):
            data_buffer.append(data_point)  # Add the current data point to the buffer
            
            # Check if the buffer is full
            if len(data_buffer) == window_size:
                # Detect anomalies within the current window
                anomalies = self.detect_anomalies(np.array(data_buffer), threshold)
                anomalies += i - window_size + 1  # Adjust indices to match original data stream
                anomaly_indices.extend(anomalies)  # Store detected anomaly indices

        # Plot the results
        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig

# Example usage:
if __name__ == "__main__":
    # Generate a sample data stream (e.g., using DataGenerator)
    generator = DataGenerator()
    simulated_data = generator.simulate_data_stream(num_points=1000, offset=5)  # Generate data with anomalies

    z_score_detector = Z_Score()  # Create an instance of the Z_Score detector
    z_score_detector.simulate_Z_score_detection(data_stream=simulated_data, window_size=50, threshold=3)  # Run anomaly detection