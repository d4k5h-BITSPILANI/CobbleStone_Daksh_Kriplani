from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator

class _IsolationForest_:
    def __init__(self, n_estimators=100, contamination=0.1):
        """
        Initializes the Isolation Forest anomaly detector.
        
        Parameters:
        - n_estimators (int): Number of base estimators in the ensemble (default is 100).
        - contamination (float): Proportion of expected anomalies in the dataset (default is 0.1).
        """
        # Validate input parameters
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not (0 < contamination < 1):
            raise ValueError("contamination must be between 0 and 1.")

        self.detector = IsolationForest(n_estimators=n_estimators, contamination=contamination)

    def train(self, X_train):
        """
        Trains the Isolation Forest model on the provided training data.
        
        Parameters:
        - X_train (np.ndarray): Training data in the shape (n_samples, n_features).
        
        Raises:
        - ValueError: If the input data is not 2D or has insufficient samples.
        """
        # Validate training data
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a numpy array.")
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if X_train.shape[0] < 2:
            raise ValueError("X_train must have at least two samples.")
        
        self.detector.fit(X_train)

    def detect_anomaly(self, new_value):
        """
        Detects if a new value is an anomaly based on the trained model.
        
        Parameters:
        - new_value (float): The new data point to check.
        
        Returns:
        - bool: True if the value is an anomaly, False otherwise.
        
        Raises:
        - ValueError: If the new_value is not a float.
        """
        # Validate new value
        if not isinstance(new_value, (float, int)):
            raise ValueError("new_value must be a numeric type (float or int).")

        is_anomaly = self.detector.predict([[new_value]])
        return is_anomaly == -1  # Returns True if it's an anomaly

    def plot_data(self, data_stream, anomaly_indices):
        """
        Plots the data stream and highlights the detected anomalies.
        
        Parameters:
        - data_stream (np.ndarray): The data stream to plot.
        - anomaly_indices (list): Indices of detected anomalies.
        
        Returns:
        - matplotlib.figure.Figure: The plotted figure.
        """
        # Validate input data for plotting
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if not isinstance(anomaly_indices, list):
            raise ValueError("anomaly_indices must be a list.")

        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Isolation Forest')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
        return fig

    def simulate_Isolation_Forest(self, data_stream):
        """
        Simulates the anomaly detection process on a given data stream.
        
        Parameters:
        - data_stream (np.ndarray): The data stream to analyze.
        
        Returns:
        - matplotlib.figure.Figure: The plotted figure with anomalies highlighted.
        
        Raises:
        - ValueError: If the input data_stream is not a numpy array or is empty.
        """
        # Validate data stream
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if data_stream.size == 0:
            raise ValueError("data_stream cannot be empty.")

        anomaly_indices = []

        # Generate synthetic training data for the model
        train_data = np.random.normal(0, 1, (1000, 1))
        self.train(train_data)  # Train the model

        # Detect anomalies in the provided data stream
        for i, value in enumerate(data_stream):
            is_anomaly = self.detect_anomaly(value)
            if is_anomaly:
                anomaly_indices.append(i)
        
        # Plot the results
        fig = self.plot_data(data_stream=data_stream, anomaly_indices=anomaly_indices)
        return fig

# Example usage:
if __name__ == "__main__":
    generator = DataGenerator()
    simulated_data = generator.simulate_data_stream(num_points=1000, offset=5)  # Generate data with anomalies

    isolation_forest_detector = _IsolationForest_()  # Create an instance of the detector
    isolation_forest_detector.simulate_Isolation_Forest(simulated_data)  # Run anomaly detection