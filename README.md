# Anomaly Detection from Continuous Data Stream

This project demonstrates various anomaly detection algorithms applied to a data stream. 
	
	1.	Data Generation: The generate_data_stream function simulates a data stream with a trend, seasonality, and noise.

	2.	Anomaly Detection Algorithms:
		1. Isolation Forest
			• Implemented in detect_anomalies_isolation_forest.
			• Description: An ensemble algorithm that isolates anomalies by creating decision trees.
			• Effectiveness: Efficient and effective for large datasets, adapts to varying data distributions, and handles both categorical and continuous features well.
		2. Z-Score
			• Implemented in detect_anomalies_z_score.
			• A statistical method that measures how many standard deviations a data point is from the mean.
			• Effectiveness: Simple and fast for identifying outliers in normally distributed data; may struggle with skewed distributions.
		3. Exponential Smoothing
			• Implemented in detect_anomalies_exponential_smoothing.
			• A forecasting method that applies weighted averages to past observations.
			• Effectiveness: Effective for time-series data with trends and seasonality, adapts to changing patterns, but may not detect abrupt changes well.
		4. Autoencoder
			• Defined by create_autoencoder and used in detect_anomalies_autoencoder.
			• A neural network that compresses and reconstructs input data, minimizing reconstruction error.
			• Effectiveness: Excellent for high-dimensional data, effectively identifies anomalies through higher reconstruction errors compared to normal data.
		
	3.	Visualization
		• The plot_data function visualizes the data stream and highlights detected anomalies.
		• Used libraries like matplotlib or Plotly to create real-time plots for data streams and anomalies.
		
	4.	Main Function:
		• It initializes a continuous data stream and applies the selected anomaly detection method.
		• The script continuously updates the plot with real-time data.
	
	5.	Optimization:
		• Implement batch processing for model fitting.
		• Use parallel processing for faster computation where applicable.
		• Minimize data copying and ensure efficient memory usage.
		• The Isolation Forest and Autoencoder models are fitted in batch mode, and parallel processing is utilized where applicable (e.g., n_jobs=-1).
		• The script employs efficient memory management by working with NumPy arrays.
	
	6. Additional Considerations:
		•	Parameter Tuning: Adjusted the parameters of each algorithm (e.g., contamination for Isolation Forest, threshold for Z-Score) based on the specific characteristics of your data.
		•	Model Complexity: The Autoencoder architecture can be adjusted (more layers or different activation functions) to suit the complexity of the data.
	

## Getting Started

### Prerequisites

Make sure you have Python 3.6 or higher installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Creating a Virtual Environment

1. **Open your terminal or command prompt.**
2. **Navigate to your project directory:**
   ```bash
   cd path/to/your/project
3.	Create a virtual environment:
   	```bash
	•	For Windows:python -m venv venv
	•	For macOS/Linux:python3 -m venv venv
5.	Activate the virtual environment:
	```bash
	•	For Windows:venv\Scripts\activate
	•	For macOS/Linux:source venv/bin/activate

Installing Requirements
	
	With the virtual environment activated, install the required packages by running: pip install -r requirements.txt
	
	Running the Main Script
	To execute the anomaly detection algorithms, run the main Python script: python main.py

Note: If you are on Windows and your Python version is 3.x, you might need to use python instead of python3.

After running the script, various anomaly detection methods will be applied to the data stream. The results will be visualized in separate plots for each algorithm:

	•	Z-Score
	•	Exponential Smoothing
	•	Isolation Forest
	•	Autoencoder
	
### Results
<img width="626" alt="Screenshot 2024-10-16 at 12 14 32 PM" src="https://github.com/user-attachments/assets/0e1a2868-e1e9-4256-8b55-9dbda976aa1f">


Each plot will display the data stream and highlight detected anomalies in red.

Deactivating the Virtual Environment 
```bash
When you’re done, you can deactivate the virtual environment by running: deactivate

