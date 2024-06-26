## Stock Price Prediction using LSTM Model and Streamlit

###Project Description

This project is designed to forecast stock prices using an LSTM (Long Short-Term Memory) model and provide a user interface for interacting with the model. The model is trained on historical stock price data and can predict prices for a specified period in the future.

###Functionality

The user interface is built using Streamlit, allowing users to conveniently interact with the stock price prediction model. The main functionality of the application includes:

**Stock Selection:** Users can choose a stock of interest from the provided list.

**Specifying Number of Days:** Users can specify the number of days for which they want to forecast the stock price.

**Getting Prediction:** After selecting the stock and specifying the number of days, users can request a forecast of the stock price. The LSTM model will be used to predict the price of the selected stock for the specified period.

### Running the Project

To run the project, follow these steps:

Install Required Libraries: Make sure you have all the necessary libraries installed as listed in the requirements.txt file.
Launch FastAPI: First, launch FastAPI, which is named main in the project. In the main.py file, you need to replace the model location with your own in the line model = pickle.load(open('Your model location model.pkl','rb')).
Launch Streamlit Application: Then, launch your Streamlit application using the command streamlit run app.py.
Using the Application
After launching, you will see the Streamlit interface where users can select a stock and specify the number of days to forecast the stock price. The LSTM prediction model will then predict the price of the chosen stock for the specified period.

###Disclaimer

Please note that while the LSTM model used in this project aims to provide accurate predictions, stock price forecasting involves inherent uncertainties and risks. The predictions provided by the model should be considered as estimates and may not always reflect the actual stock prices. The developer of this project assumes no responsibility for any financial decisions made based on the predictions generated by the model. Users are advised to conduct their own research and consult with financial professionals before making any investment decisions.

##Additions to the Project

Connected backend on FastAPI: The project has a backend connected on FastAPI, where the model is deployed using pickle and all calculations are performed.
Improved User Interface: The user interface has been improved, in particular, a block has been added where closing price positions are visible in turn.
