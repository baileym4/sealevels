# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Bailey McIntyre
# Collaborators: None / Office Hours

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        # NOTE: TO BE IMPLEMENTED IN PART 4B.2 OF THE PSET
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.
    
        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at
    
        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        
        
        year_averages = []
        for year in years:
            temps = []
            for city in cities:
                temp_1 = self.get_daily_temps(city, year)
                temp_1_mean = np.mean(temp_1)
                temps.append(temp_1_mean)
            temp = np.array(temps) 
            year_avg = np.mean(temp)
            year_averages.append(year_avg)
                
        return year_averages 

    
        

def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    
    # x coord is the year 
    # y coord is the temp
    
    # first find average values
    
    y_avg = np.mean(y)
    x_avg = np.mean(x)
    
    # now calculate slope
    
    numerator = 0 
    denominator = 0  
    
    for i in range(len(x)):
        numerator += (x[i] - x_avg) * (y[i] - y_avg)
        denominator += (x[i] - x_avg) * (x[i] - x_avg)
    
    slope = numerator / denominator 
    intercept = y_avg - (slope * x_avg)
    
    final_tup = (slope, intercept)
    
    return final_tup
                



def squared_error(x, y, m, b):
    """
   Calculates the squared error of the linear regression model given the set
   of data points.

   Args:
       x: a 1-d numpy array of length N, representing the x-coordinates of
           the N sample points
       y: a 1-d numpy array of length N, representing the y-coordinates of
           the N sample points
       m: The slope of the regression line
       b: The y-intercept of the regression line


   Returns:
       a float for the total squared error of the regression evaluated on the
       data set
   """
   
       # squared errror is sum of y - yestimate sqaures
       
    y_vals_regress = []
       
    for val in x:
        y_val_new = (m * val) + b
        y_vals_regress.append(y_val_new)
         
    error = 0 
    
    for i in range(len(y)):
        error += (y[i] - y_vals_regress[i]) ** 2
        
    return error
   
    


def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
        
        The models should appear in the list in the same order as their corresponding 
        integers in the `degrees` parameter
    """
    coefficent_lists = []
    for degree in degrees:
        model = np.polyfit(x, y, degree)   
        coefficent_lists.append(model)
        
    return coefficent_lists
        


def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve. You should make a separate plot for each model.
    
    Your plots should adhere to the following guidelines:

        - Plot the data points as individual green (color='C2') dots.
        - Plot the model with an orange (color='C1') solid line.
        - Include a title. Your title should include the $R^2$ value of the model and the degree. If the model is a linear curve (i.e. its degree is one), the title should also include the ratio of the standard error of this fitted curve's slope to the slope. Round your $R^2$ and SE/slope values to 4 decimal places.
        - Label the axes. You may assume this function will only be used in the case where the x-axis represents years and the y-axis represents temperature in degrees Celsius.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    
    # use page 41 from lecture slides to create plots 
    
    y_predicted = [] # lists of fit y values 
    
    # find predicted y vals based on models
    
    for model in models:
        y_est = np.polyval(model, x)
        y_predicted.append(y_est)
    
    r_squared = [] # list of R^2 values
    
    # calculate R^2 for each model
    
    for prediction in y_predicted:
        error = ((prediction - y) ** 2).sum() # took this from lecture slides 
        mean_error = error / len(y)
        r_sq = 1 - (mean_error / np.var(y))
        r_squared.append(r_sq)
    
        
  
    if display_graphs:
        for i in range(len(models)):
            #plt.cla()  ## to clear plots (?)
           # y_model = np.polyval(x, model[i])
            plt.plot(x, y_predicted[i], color='C1')
            plt.scatter(x, y, s=20, c='C2')
           # title = "Degree value " + str(models[i].size - 1) + " with R^2 value " + str(round(r_squared[i], 4))
            if (models[i].size - 1) == 1:
                standard_error = standard_error_over_slope(x, y, y_predicted[i], models[i])
                #title = "Degree value " + str(models[i].size - 1) + " with R^2 value " + str(round(r_squared[i], 4)) + " and ratio of standard error of fitted curve to slope " + str(round(standard_error, 4)) # divide by slope of line models[0]
                plt.title(f"Plot of data sample and fitted curve. R\u00b2 = {r_squared[i]:.4f}, Degree = {models[i].size - 1}, SE/slope = {standard_error:.4f}")
            else:
                plt.title(f"Plot of data sample and fitted curve. R\u00b2 = {r_squared[i]:.4f}, Degree = {models[i].size - 1}")
            plt.xlabel("Years")
            plt.ylabel("Temperature (Celsius)")
           
            plt.show() # show up plot into for loop
       
            
  
    
    
    return r_squared
    


def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    
    slopes = {} # key is tuple of i, j and value is the slope
    total_length = len(x)
    
    for val in range(len(x)):
        final = val + length
        if final <= total_length:
            current_tup = (val, final)
            current_slope = linear_regression(x[val:final], y[val:final])[0]  
            slopes[current_tup] = current_slope
            
    ranges = list(slopes.keys())
    slopes_ranges = list(slopes.values())        
    
    

    if positive_slope:
        max_slope = float('-inf')
        for m in slopes_ranges:
            if m > max_slope + (10 ** -8):
                max_slope = m
                
        
    else:
        max_slope = float('inf')
        for m in slopes_ranges:
            if m < max_slope - (10 ** -8):
                max_slope = m
    
    
    
    if positive_slope:
        if max_slope < 0:
            return None
    else:
        if max_slope > 0:
            return None
    
    
    
    index_needed = slopes_ranges.index(max_slope)   # does this account for ties (?)
    
    i, j = ranges[index_needed] 
 
    final_tup = (i, j, max_slope)
    
    return final_tup
            
    
    #raise NotImplementedError


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    final_list = []
    for i in range(2, len(x)+1): # include last
        ans_p = get_max_trend(x, y, i, True)
        #i_p, j_p, most_positive = ans_p[0], ans_p[1], ans_p[2]
        ans_n = get_max_trend(x, y, i, False)
        #i_n, j_n, most_negative = ans_n[0], ans_n[1], ans_n[2] # none type not suscriptable (?)
        if ans_p == None and ans_n == None:
            current_tup = (0, i, None)
        
        elif ans_p == None:
            current_tup = ans_n
        elif ans_n == None:
            current_tup = ans_p
        
        elif abs(ans_p[2]) > abs(ans_n[2]):
            current_tup = ans_p
        
        else:
            current_tup = ans_n
        
        
        final_list.append(current_tup)
    
    return final_list
            
    
    


def calculate_rmse(y, estimated):
    
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    numerator_sum = 0
    denominator = len(y)
    for i in range(len(y)):
        numerator_sum += (y[i] - estimated[i]) ** 2
        
    rmse = (numerator_sum / denominator) ** (1/2)
    
    return rmse
    
    


def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
        """
    # find the estimated y vals for given model
    estimations = []
    for model in models:
        estimated = np.polyval(model, x)
        estimations.append(estimated)
        
        
        
    rmse_vals = [] # list of rmse vals

    
    # find rmse for each model
    for estimation in estimations:
        rmse_current = round(calculate_rmse(y, estimation), 4)
        rmse_vals.append(rmse_current)
     
    if display_graphs:
        for i in range(len(models)):
            plt.plot(x, estimations[i], color='r')
            plt.scatter(x, y, s=20, c='blue')
            title = "Degree Value " + str(models[i].size - 1) + " with RMSE " + str(rmse_vals[i])
            plt.title(title)
            plt.xlabel("Years")
            plt.ylabel("Temperature (Celsius")
            plt.show()
         
   
    
    
    return rmse_vals
    #raise NotImplementedError


if __name__ == '__main__':
    
    
    
    #pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    data = Dataset("data.csv")
    
    years = range(1961, 2017) # want to include 2017 (??? unclear)
    temp_info = []
    for year in years:
        temp = data.get_temp_on_date("SAN FRANCISCO", 12, 1, year)
        temp_info.append(temp)
    
    # make arrays 
    years = np.array(years)
    temps = np.array(temp_info)
    
    # generate polynomial results 
    # evaluate models 
    degrees = [1]
    degrees = np.array(degrees)  # want model that shows every year not each indiivual year 
    models = generate_polynomial_models(years, temps, degrees) # get models 
    
    evaluate_models(years, temps, models, True) # plot models  
    
    
    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    
    cities = ["SAN FRANCISCO"]
    years_avg = np.array(range(1961, 2017))
    avg_annual = data.calculate_annual_temp_averages(cities, years)
    
    mean_annual = np.array(avg_annual)
    models_ = generate_polynomial_models(years_avg, mean_annual, degrees) # generate models
    
    evaluate_models(years_avg, mean_annual, models_, True)
    

    ##################################################################################
    # Problem 5B: INCREASING TRENDS

    years_total = np.array(range(1961, 2017))
    temps_per_year = data.calculate_annual_temp_averages(["SEATTLE"], years_total)
    year_start, year_end, slope = get_max_trend(years_total, temps_per_year, 30, True)
    year_range = np.array(years_total[year_start: year_end])
    temps_range = np.array(temps_per_year[year_start: year_end])
    
    modelss = generate_polynomial_models(year_range, temps_range, degrees)
    evaluate_models(year_range, temps_range, modelss, True)
    
    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    
    
    year_start_n, year_end_n, slope_n = get_max_trend(years_total, temps_per_year, 12, False)
    year_range_n = np.array(years_total[year_start_n: year_end_n])
    temps_range_n = np.array(temps_per_year[year_start_n: year_end_n])
    
    models_n = generate_polynomial_models(year_range_n, temps_range_n, degrees)
    evaluate_models(year_range_n, temps_range_n, models_n, True)

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.
    
    

    ##################################################################################
    # Problem 6B: PREDICTING
    
    #### generating models based on training data
    
    # what cities should we be using ? cities at start (?)
    training_data = np.array(TRAIN_INTERVAL)
    #train = np.array(training_data)
    avg_test = data.calculate_annual_temp_averages(CITIES, training_data)
    #degrees_train = [2, 10]
    
    models_p = generate_polynomial_models(training_data, avg_test, [2, 10])
    r_sq = evaluate_models(training_data, avg_test, models_p, True)
    
    #### predicting data 
    testing_data = np.array(TEST_INTERVAL)
    avg_predicted = data.calculate_annual_temp_averages(CITIES, testing_data)
    evaluate_rmse(testing_data, avg_predicted, models_p, True)
    
    
    ####################################################################################








