# CROP-RECOMMENDATION-SYSTEM
The Crop Recommendation System is a machine learning project designed to assist farmers in selecting the most suitable crop for their fields based on various environmental and soil parameters. By leveraging machine learning algorithms, the system analyzes the input data to recommend crops that are most likely to thrive, thereby optimizing yield and resources.

## Features
- Accepts soil and environmental parameters as input (e.g., nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall).
- Provides crop recommendations based on the input data.
- User-friendly interface for farmers and agricultural experts.
- Scalable and adaptable for diverse agricultural regions.

## Technologies Used
- Programming Language: Python
- Libraries:
  - Machine Learning: Scikit-learn, NumPy, Pandas
  - Data Visualization: Matplotlib, Seaborn
- Model Used: Random Forest Classifier, SVM, KNN
- Dataset: Crop recommendation dataset from kaggle

## Installation

### Prerequisites
- Python 3.8 or later
- Virtual environment (recommended)
- Required libraries (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sathwikavardhineedi/CROP-RECOMMENDATION-SYSTEM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd crop-recommendation-system
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Usage
1. Launch the application by running the `app.py` file.
2. Enter the required soil and environmental parameters.
3. Click on the Submit button to get crop recommendations.
4. View the recommended crops on the screen.

## Dataset
- Source: Kaggle
- Attributes:
  - Nitrogen, Phosphorus, Potassium
  - Temperature, Humidity, pH, Rainfall
  - Target: Crop type

## Results
- Accuracy:99% 
- Performance Metrics: Precision, Recall, F1-Score

## Future Work
- Enhance the model with additional datasets for improved accuracy.
- Develop a mobile application for easier access.
- Integrate real-time weather data.
- Add multi-language support for global usability.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
