import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from data_generator import DataGenerator

class Autoencoder:
    def __init__(self, input_dim, encoding_dim=8):
        """
        Initializes the Autoencoder model.
        
        Parameters:
        - input_dim (int): Dimension of the input data.
        - encoding_dim (int): Dimension of the encoded representation (default is 8).
        """
        # Validate input parameters
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if encoding_dim <= 0:
            raise ValueError("encoding_dim must be a positive integer.")
        
        # Define the autoencoder structure
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)  # Encoding layer
        decoded = Dense(input_dim, activation='sigmoid')(encoded)       # Decoding layer
        
        self.autoencoder = Model(input_layer, decoded)  # Create the autoencoder model
        self.autoencoder.compile(optimizer='adam', loss='mse')  # Compile the model

    def train(self, X_train, epochs=10, batch_size=32):
        """
        Trains the autoencoder on the provided training data.
        
        Parameters:
        - X_train (np.ndarray): Training data (shape must match input_dim).
        - epochs (int): Number of training epochs (default is 10).
        - batch_size (int): Number of samples per gradient update (default is 32).
        
        Raises:
        - ValueError: If X_train is not a numpy array or does not have the correct shape.
        """
        # Validate training data
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a numpy array.")
        if X_train.shape[1] != self.autoencoder.input_shape[1]:
            raise ValueError(f"X_train must have shape (n_samples, {self.autoencoder.input_shape[1]}).")

        # Train the autoencoder
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    def detect_anomaly(self, new_value):
        """
        Detects anomalies by reconstructing the input value and calculating the reconstruction loss.
        
        Parameters:
        - new_value (np.ndarray): New data point to check for anomalies.
        
        Returns:
        - bool: True if the reconstruction loss exceeds the threshold, indicating an anomaly.
        
        Raises:
        - ValueError: If new_value is not a numeric type.
        """
        # Validate new value
        if not isinstance(new_value, (np.ndarray, list, float, int)):
            raise ValueError("new_value must be a numeric type (numpy array, list, float, or int).")

        # Reshape new_value to match model input
        new_value = np.array(new_value).reshape(1, -1)
        
        # Predict the reconstructed value
        reconstructed_value = self.autoencoder.predict(new_value)
        
        # Calculate the mean squared error as the loss
        loss = np.mean(np.square(new_value - reconstructed_value))
        
        # Determine if the loss indicates an anomaly
        return loss > 0.1  # Threshold for anomaly detection

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
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if not isinstance(anomaly_indices, (list, np.ndarray)):
            raise ValueError("anomaly_indices must be a list or numpy array.")

        fig = plt.figure()
        plt.plot(data_stream, label='Data Stream')
        plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
        plt.title('Anomaly Detection using Autoencoder')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
        return fig

    def simulate_autoencoder(self, data_stream):
        """
        Simulates the anomaly detection process on a given data stream.
        
        Parameters:
        - data_stream (np.ndarray): The data stream to analyze.
        
        Returns:
        - matplotlib.figure.Figure: The plotted figure with anomalies highlighted.
        
        Raises:
        - ValueError: If data_stream is not a numpy array or is empty.
        """
        # Validate data stream
        if not isinstance(data_stream, np.ndarray):
            raise ValueError("data_stream must be a numpy array.")
        if data_stream.size == 0:
            raise ValueError("data_stream cannot be empty.")

        anomaly_indices = []  # List to store indices of detected anomalies
        
        # Generate synthetic training data for the model
        train_data = np.random.normal(0, 1, (1000, 1))  # 1000 samples of normal data
        input_dim = train_data.shape[1]  # Get the input dimension
        self.train(train_data, epochs=10, batch_size=32)  # Train the model

        # Detect anomalies in the provided data stream
        for i, value in enumerate(data_stream): 
            new_value = value  # Current data point
            is_anomaly = self.detect_anomaly(new_value)  # Check for anomaly
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

    autoencoder_detector = Autoencoder(input_dim=1)  # Create an instance of the Autoencoder
    autoencoder_detector.simulate_autoencoder(data_stream=simulated_data)  # Run anomaly detection