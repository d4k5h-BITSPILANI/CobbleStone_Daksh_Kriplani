import numpy as np
import matplotlib.pyplot as plt

from data_generator import DataGenerator
from z_score import Z_Score
from exponential_smoothing import ExponentialSmoothing
from isolation_forest import _IsolationForest_
from autoencoder import Autoencoder

def main():
    """
    Main function to simulate anomaly detection using various algorithms.
    
    This function generates a data stream, applies different anomaly detection
    algorithms (Z-Score, Exponential Smoothing, Isolation Forest, and Autoencoder),
    and visualizes the results.
    """
    # Parameters for data simulation
    num_points = 1000  # Total number of data points to generate
    offset = 5  # Offset value to introduce anomalies
    
    # Create a DataGenerator instance and simulate a data stream
    data_gen = DataGenerator()
    data_stream = data_gen.simulate_data_stream(num_points=num_points, offset=offset)
    
    # Parameters for Z-Score detection
    window_size = 50  # Size of the sliding window for Z-Score
    threshold = 3  # Z-Score threshold for anomaly detection

    # Initialize Z-Score anomaly detector
    z_sc = Z_Score()
    fig_z_sc = z_sc.simulate_Z_score_detection(data_stream, window_size, threshold)  # Run Z-Score detection
    
    # Initialize Exponential Smoothing anomaly detector
    exponen_sc = ExponentialSmoothing()
    fig_exponen_sc = exponen_sc.simulate_Exponential_Smoothing(data_stream=data_stream)  # Run Exponential Smoothing detection

    # Initialize Isolation Forest anomaly detector
    isolation_forest = _IsolationForest_()
    fig_isolation_forest = isolation_forest.simulate_Isolation_Forest(data_stream=data_stream)  # Run Isolation Forest detection

    # Initialize Autoencoder anomaly detector
    input_dim = 1  # Input dimension for the Autoencoder
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=8)
    fig_autoencoder = autoencoder.simulate_autoencoder(data_stream=data_stream)  # Run Autoencoder detection

    # Display all plots
    plt.show()

if __name__ == "__main__":
    main()