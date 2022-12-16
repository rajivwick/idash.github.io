import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from scalecast.Forecaster import Forecaster
from tensorflow.keras.callbacks import EarlyStopping




def py_bar(item_df,item):

# Set date column as index of imported item_df
    item_df = item_df.set_index('date')
# Store sold column data
    line_plot = item_df.sold

# Initialize plot size
    fig, ax = plt.subplots(figsize=(10,10))

# Group by year and month
    bar_plot = line_plot.groupby([line_plot.index.year, line_plot.index.month]).mean().unstack()
    bar_plot.plot(ax=ax, kind='bar', width=0.8)
    ax.set_xlabel('Years')
    ax.set_ylabel('Items Sold (Montly)')
    
# Label for legend | Displaying months 
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [dt.date(1900, int(monthinteger), 1).strftime('%B') for monthinteger in labels]
    ax.legend(handles = handles, labels = new_labels, loc = 'upper left', bbox_to_anchor = (1.02, 1))
    
# Add Title
    plt.title(item)
    
# Remove spacing from item name and replacing it with an underscore - required to allow bootstrap to load images.
    item = item.replace(r' ', '_')  
    
# Set up save location
    savefile = "static/images/" + item + "_bar" + ".png"
    
# Save graph
    plt.savefig(savefile, bbox_inches='tight',pad_inches=0.2, transparent=True)
    

# Current in-active module - can be used to generate useful scatterplot - intention for POS item.
def sea_scatter(item_df,item):

# Initialize plot size    
    fig_size = (12,8)
    ax = plt.subplots(figsize=fig_size)

# Create scatterplot with parameters
    g=sns.scatterplot(ax=ax,data=item_df,
                    x="date", 
                    y="sold",
                    markers=True,
                    style='Month', 
                    hue="Month", 
                    size='sold', 
                    legend='full', 
                    palette="muted")
    g.set_ylabel('Total Sold',fontsize=16)
    g.set_xlabel('Date',fontsize=16)

# Label for legend | Displaying months 
    h,l = g.get_legend_handles_labels()
    plt.legend(h[0:13],l[0:13],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=13)
# Add Title
    plt.title(item,fontsize=20)
# Remove spacing from item name and replacing it with an underscore - required to allow bootstrap to load images.
    item = item.replace(r' ', '_')  
# Set up save location
    savefile = "static/images/" + item + "_scatter" + ".png"
# Save graph
    plt.savefig(savefile,bbox_inches='tight',pad_inches=0.2, transparent=True)

   

def sql_meat_item(session,Meat,type):

# Initialize query calls from meat_historical_prices table 
    query = [Meat.date, Meat.price, Meat.percentage_change]

# Query the SQLite database connected for all data relevant to the query layed out above relating to the type of meat in question.
    item_query = session.query(*query).\
    filter_by(type=type).all()

# Store query results as a Dataframe and set column names    
    item_df = pd.DataFrame(item_query, columns=['date', 'price', 'percentage_change'])

# Convert the date column to datetime type    
    item_df['date'] = pd.to_datetime(item_df['date'], format="%b-%y")

# Sort the Dataframe by the date column    
    item_df = item_df.sort_values(by=['date'])
    return item_df

def sql_total(session, Sales):

# Initialize query calls from sales table     
    query = [Sales.item, Sales.sold, Sales.date, Sales.tot_value]

# Query all data in the sales table relating to the query above    
    item_query = session.query(*query).all()

# Store query results as a Dataframe and set column names
    item_df = pd.DataFrame(item_query, columns=['item', 'sold', 'date', 'total($)'])

# Convert the date column to datetime type     
    item_df['date'] = pd.to_datetime(item_df['date'], format='%Y/%m/%d')

# Sort the Dataframe by the date column      
    item_df = item_df.sort_values(by=['date'])

# Clean Dataframe - drop duplicates and NaN values
    item_df = item_df.drop_duplicates()
    item_df = item_df.dropna()
        
    return item_df

def sql_item(session,Sales,item):

# Initialize query calls from sales table 
    query = [Sales.item, Sales.sold, Sales.date, Sales.tot_value]

# Query the SQLite database connected for all data relevant to the query layed out above relating to the type of meat in question.
    item_query = session.query(*query).\
    filter_by(item=item).all()

# Store query results as a Dataframe and set column names    
    item_df = pd.DataFrame(item_query, columns=['item', 'sold', 'date', 'total($)'])

# Convert the date column to datetime type     
    item_df['date'] = pd.to_datetime(item_df['date'], format='%Y/%m/%d')

    item_df = item_df.sort_values(by=['date'])
# Sort the Dataframe by the date column       
    return item_df

def ml_engine(item_df,item,column,interval):
    
# Initialize Forcaster with Y and X (Current dates)
    f = Forecaster(y=item_df[column],
                    current_dates=item_df['date'])
            
    f.set_test_length(12)       # 1. 12 observations to test the results
    f.generate_future_dates(12) # 2. 12 future points to forecast
    f.set_estimator('lstm')     # 3. LSTM neural network

# First standard LSTM model - bare minimum
    f.manual_forecast(call_me='lstm_default')
    
# Second LSTM model - lags 24
    f.manual_forecast(call_me='P1',lags=24)
    
# Third LSTM model - lags 24, epochs 5
    f.manual_forecast(call_me='P2',
                    lags=24,
                    epochs=5,
                    validation_split=.2,
                    shuffle=True)
    
# Fouth LSTM model - lags 24, epochs 36, lstm layer size = 18, 18 and 18  
    f.manual_forecast(call_me='P3',
                    lags=24,
                    epochs=36,
                    validation_split=.2,
                    shuffle=True,
                    callbacks=EarlyStopping(monitor='val_loss',
                                                patience=5),
                    lstm_layer_sizes=(18,18,18),
                    dropout=(0,0,0))

# Fifth LSTM model - lags 36, epochs 32, lstm layer size = 72*4, activation = tanh, optimizer = Adam       
    f.manual_forecast(call_me='P4',
                    lags=36,
                    batch_size=32,
                    epochs=16,
                    validation_split=.2,
                    shuffle=True,
                    activation='tanh',
                    optimizer='Adam',
                    learning_rate=0.001,
                    lstm_layer_sizes=(72,)*4,
                    dropout=(0,)*4,
                    plot_loss=False)
   
# Set mlr parameters    
    f.set_estimator('mlr') 
    f.add_ar_terms(24) 

# Add seasonal regressor parameters
    f.add_seasonal_regressors('month',dummy=True) 
    f.add_seasonal_regressors('year') 
    f.add_time_trend() 
    f.diff() 
    f.manual_forecast()

# Plot final prediction graph displaying only the top outcome
    f.plot(models='top_1',
            order_by='LevelTestSetMAPE',
            level=True)

# Set graph y-axis label and title         
    plt.ylabel('Count')
    plt.title(item)

# Set interval of x-axis ticks and display them as mm-yyyy    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.gcf().autofmt_xdate() # label Rotation

# Remove spacing from item name and replacing it with an underscore - required to allow bootstrap to load images. 
#    
    item = item.replace(r' ', '_')
# Save file name  and location       
    savefile = "static/images/" + item + "_pred" +'-'+ column + ".png"

# Save plot to location
    plt.savefig(savefile,bbox_inches='tight',pad_inches=0.2, transparent=True)
    
# Output the results from Forecaster to an excel file containing all necessary information from model analysis.
    ModelScoresName = 'Results/'+ item + "_modelscore" +'-'+ column + ".xlsx"
    f.export(
                    to_excel=True,
                    determine_best_by='LevelTestSetMAPE',
                    excel_name= ModelScoresName 
            )

def sql_engine(include_items):

# Set sqlite database name and input into the create_engine model from sqlalchemy 
    engine = create_engine("sqlite:///RestaurantDB.sqlite")

# Use automap_base to auto map the Base classes found within the database
    Base = automap_base()
    Base.prepare(engine, reflect=True)

# Create class assignments for required tables within the database
    Sales = Base.classes.sales
    Meat = Base.classes.meat_historical_prices

# Create a session with SQLite
    session = Session(engine)

# Query designed to deliver a descending array of item and their total sold counts
    #query2 = [Sales.item, func.sum(Sales.sold)]

        #total_sold = session.query(*query2).\
        #group_by(Sales.item).all()

        #total_sold_df = pd.DataFrame(total_sold, columns=['item', 'quantity']).\
        #    sort_values(by='quantity', ascending=False).reset_index(drop=True)

# Store revenue data by using sql_total function
    revenue_df = sql_total(session, Sales)

# Group by dates and count total revenue - apply some data cleaning and reset index
    revenue_df=revenue_df.groupby(revenue_df['date'].dt.strftime('%Y-%m')).sum().reset_index()

# Initializae required fields and operate ML models
    name = 'Revenue'
    column = 'total($)'
    ml_engine(revenue_df,name,column,3)

# Initialize item_df
    item_df = pd.DataFrame()

# Created ignore_items as precursor for development with automated top and bottom 5-10 items analysis allowing for user to disregard any items of their choosing.    
    #ignore_items = ['Paper Carry Bag','BYO Per Head','Reuserble Bag', 'Can drinks']


    
    
    for item in include_items:

        # Used in conjunction with automated item analysis feature
        #item = total_sold_df.loc[item,'item']

        item_df = sql_item(session,Sales,item)

        # Clean Dataframe
        item_df['Month'] = item_df['date'].dt.month
        item_df = item_df.drop_duplicates('date')
        
        #if item not in ignore_items:
        #    item_df = sql_item(session,Sales,item)
        #    item_df['Month'] = item_df['date'].dt.month
        #    item_df = item_df.drop_duplicates('date')
        

        column = 'sold'
        
        # Run ML model anaylsis 
        ml_engine(item_df, item, column,3)

        # Generate py bar graph
        py_bar(item_df,item)

        
    meat_list = ['beef', 'chicken']
    nomalisation = ['minmax', 'scaler', 'none']

    meat_df = pd.DataFrame()
    for protein in meat_list:
        meat_df = sql_meat_item(session,Meat,protein)
        for style in nomalisation:
            if style == 'minmax':
                data = meat_df['price']
                values = data.values
                values = values.reshape((len(values), 1))
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler = scaler.fit(values)
                normalized = scaler.transform(values)
                normalized = pd.DataFrame(normalized)
                meat_df[style] = normalized
                ml_engine(meat_df, protein, style,12)
            
            elif style == 'scaler':
                data = meat_df['price']
                values = data.values
                values = values.reshape((len(values), 1))
                scaler = StandardScaler()
                scaler = scaler.fit(values)
                normalized = scaler.transform(values)
                normalized = pd.DataFrame(normalized)
                meat_df[style] = normalized
                ml_engine(meat_df, protein, style,12)
            
            elif style == 'none':
                none = 'price'
                ml_engine(meat_df, protein, none, 12)