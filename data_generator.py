import numpy as np

class DataGenerator:
    def __init__(self) -> None:
        """
        Initializes the DataGenerator class.
        Currently, there are no attributes to initialize, but the constructor
        is in place for potential future enhancements or attributes.
        """
        pass
    
    def simulate_data_stream(self, num_points=1000, offset=5):
        """
        Simulates a continuous data stream of floating-point numbers 
        with random anomalies.
        
        Parameters:
        - num_points (int): The total number of data points to generate.
          Default is 1000.
        - offset (float): The amount added to the randomly chosen 
          anomaly points to make them deviate from the normal distribution. 
          Default is 5.

        Returns:
        - np.ndarray: An array representing the simulated data stream, 
          including normal values and introduced anomalies.
        """
        # Set a random seed for reproducibility of results
        np.random.seed(42)
        
        # Generate a normally distributed data stream with mean=0 and std_dev=1
        data_stream = np.random.normal(0, 1, num_points)
        
        # Randomly select 10 unique indices from the data stream to introduce anomalies
        anomaly_indices = np.random.choice(num_points, size=10, replace=False)
        
        # Introduce anomalies by adding an offset to the selected indices
        data_stream[anomaly_indices] += offset 
        
        return data_stream  # Return the generated data stream with anomalies

# Example usage
if __name__ == "__main__":
    generator = DataGenerator()  # Create an instance of the DataGenerator class
    simulated_data = generator.simulate_data_stream(num_points=1000, offset=5)  # Simulate the data stream
    print(simulated_data)  # Print the generated data stream