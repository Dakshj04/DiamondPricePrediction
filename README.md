# 💎 Diamond Price Prediction

A comprehensive machine learning project that predicts diamond prices based on various physical characteristics using a Flask web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning pipeline to predict diamond prices based on key characteristics such as carat weight, cut quality, color grade, clarity, and physical dimensions. The system includes:

- **Data Processing Pipeline**: Automated data ingestion, transformation, and preprocessing
- **Model Training**: Multiple regression models with automatic hyperparameter optimization
- **Web Interface**: User-friendly Flask application for real-time predictions
- **Model Persistence**: Serialized models for production deployment

## ✨ Features

- **Multi-Model Comparison**: Tests Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, and K-Neighbors
- **Automated Pipeline**: End-to-end data processing and model training
- **Web Interface**: Interactive form for diamond price predictions
- **Model Persistence**: Saves trained models and preprocessors for reuse
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Error Handling**: Robust exception handling and custom error classes

## 📁 Project Structure

```
Diamond Price Prediction/
├── application.py              # Main Flask application
├── requirements.txt            # Python dependencies
├── setup.py                   # Package configuration
├── README.md                  # Project documentation
├── artifacts/                 # Saved models and data
│   ├── model.pkl             # Trained model
│   ├── preprocessor.pkl       # Data preprocessor
│   ├── raw.csv               # Original dataset
│   ├── train.csv             # Training data
│   └── test.csv              # Test data
├── src/                      # Source code
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipelines/            # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils.py              # Utility functions
│   ├── logger.py             # Logging configuration
│   └── exception.py          # Custom exception handling
├── templates/                # HTML templates
│   ├── index.html           # Home page
│   └── form.html            # Prediction form
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── Model.ipynb         # Model development
│   └── data/
│       └── gemstone.csv    # Original dataset
└── logs/                   # Application logs
```

This comprehensive README file provides:

1. **Clear project overview** with features and capabilities
2. **Detailed project structure** showing the organization of files and directories
3. **Installation instructions** with step-by-step setup
4. **Usage guidelines** for both training and prediction
5. **Technical architecture** explaining the ML pipeline
6. **API documentation** for the web interface
7. **Contributing guidelines** for potential collaborators

The README is structured to be both informative for developers and accessible for users who want to understand and use the diamond price prediction system.

## ️ Technologies Used

- **Python 3.x**
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Pickle** - Model serialization

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Diamond-Price-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## 📖 Usage

### Training the Model

1. **Run the training pipeline**
   ```bash
   python src/pipelines/training_pipeline.py
   ```

   This will:
   - Load and preprocess the data
   - Split into training and test sets
   - Train multiple models
   - Select the best performing model
   - Save the model and preprocessor

### Running the Web Application

1. **Start the Flask application**
   ```bash
   python application.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - Click on "Predict" to access the prediction form
   - Fill in the diamond characteristics
   - Submit to get the predicted price

### Making Predictions

The web interface accepts the following diamond characteristics:

- **Carat**: Weight of the diamond (0.1 - 5.0)
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color**: Color grade (D, E, F, G, H, I, J)
- **Clarity**: Clarity grade (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- **Depth**: Depth percentage (40-80)
- **Table**: Table percentage (40-95)
- **X, Y, Z**: Length, width, and height in mm

## 🏗️ Model Architecture

### Data Pipeline

1. **Data Ingestion** (`data_ingestion.py`)
   - Loads data from CSV files
   - Splits into training and test sets (70/30 split)
   - Saves processed datasets

2. **Data Transformation** (`data_transformation.py`)
   - Handles categorical variables (cut, color, clarity)
   - Applies feature scaling
   - Creates preprocessing pipeline

3. **Model Training** (`model_trainer.py`)
   - Tests multiple regression models:
     - Linear Regression
     - Lasso Regression
     - Ridge Regression
     - ElasticNet
     - Decision Tree
     - Random Forest
     - K-Neighbors
   - Evaluates using R² score
   - Selects best performing model

### Prediction Pipeline

1. **Custom Data Class** - Validates and formats input data
2. **Preprocessing** - Applies the same transformations as training
3. **Prediction** - Uses the trained model to predict prices

##  API Endpoints

- `GET /` - Home page
- `GET /predict` - Prediction form
- `POST /predict` - Submit prediction request

## 📊 Model Performance

The system automatically evaluates multiple models and selects the best performer based on R² score. The trained model is saved in `artifacts/model.pkl` and can be used for production predictions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ‍💻 Author

**Daksh** - [dakshcdr@gmail.com](mailto:dakshcdr@gmail.com)

---

**Note**: This project is for educational and demonstration purposes. For production use, additional considerations such as model validation, monitoring, and security should be implemented.