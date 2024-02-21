# Temperature Analysis Tool

## Overview

This Python script serves as a data analysis tool for temperature records. It includes functionalities for polynomial regression modeling, trend analysis, and prediction of temperature data. The script is designed to work with temperature data provided in the `data.csv` file.

## Contents

1. **File Structure**
   - `temperature_analysis.py`: Main script containing functions for data analysis.
   - `data.csv`: CSV file containing temperature records.
   - `test_ps5_student.py`: Test cases for the script, provided by the 6.100B MIT course staff.

2. **Dependencies**
   - NumPy
   - Matplotlib
   - scikit-learn

3. **Usage**

   To run the script, follow these steps:

   ```bash
   git clone https://github.com/your-username/temperature-analysis.git
   cd temperature-analysis
   python temperature_analysis.py

## Functionalities

1. **Daily Temperature Analysis:**
   - The script provides polynomial regression models for daily temperature data, with the option to visualize the results.

2. **Annual Temperature Analysis:**
   - It calculates and visualizes the average annual temperatures for specified cities over the years.

3. **Trend Analysis:**
   - The script identifies intervals with extreme positive or negative slopes, showcasing trends in the temperature data. Functions exist to calculate the mse and rmse

4. **Prediction:**
   - Using polynomial regression models, the script predicts future temperature trends based on training data.

## Test Cases

- Test cases are available in `test_ps5_student.py`, which is written by the MIT 6.100B course staff. These tests ensure the correctness and functionality of the implemented code.


## Acknowledgments

- This script was developed as part of the 6.100B MIT course.
- The temperature data is sourced from the `data.csv` file.

