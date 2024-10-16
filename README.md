# Anomaly Detection from Continuous Data Stream

This project demonstrates various anomaly detection algorithms applied to a data stream. 

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

