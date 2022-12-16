
import pandas as pd
import scripts.ETL as ETL ## Extract, Transform, Load - python script
import scripts.pytrend as py ## pytrends query and plotting - python script
import scripts.sql_queries_ml as ml ## machine learning script as well as plotting of relevant graphs
import datetime as dt

from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def index():
    
    nowDate = dt.datetime.now().date()  
    
    # Generate pytrends information
    # Key Word selector
    kw_list = ["indian food"]

    #Initalize Key Word Interest Over Time Dataframe
    kwIOT_df = pd.DataFrame()

    #Call pytrends_data module from script
    kwIOT_df = py.pytrends_data(kw_list, kwIOT_df)

    #Store index (dates) of kwIOT_df as a variable
    kw_index = kwIOT_df.index

    #Plot the Interest Over Time of keywords related to the keyword
    py.sea_scatter(kwIOT_df,kw_index)

    # ETL Functionality 
    Starting_year = 17
    Ending_year = 22

    #initialize master_df to store all POS data
    master_df = pd.DataFrame()

    # Extract and Transform  - POS data  
    master_df = ETL.Load_master_df(master_df,Starting_year,Ending_year)

    # Initialize meat_master_df to store all meat historical data
    meat_master_df = pd.DataFrame()
    # Target File Names
    filename = ['beef', 'chicken']

    # Initialize database name
    dbname = 'RestaurantDB'

    # Extract and Transform - meat historical data
    meat_master_df = ETL.Load_meat_master_df(meat_master_df,filename)

    # Load transformed data into SQLite
    ETL.CreateDB(dbname, master_df, meat_master_df)

    # Hard coded items to generate predictions and visual analysis on - purpose to reduce load time as generating the top 5-10 as initially planned would consume a large amount of time.
    # For a finalized version of this dashboard, a goal would be to run this app with auto generation of top 5-10 items and/or bottom 5-10 itmes.
    
    include_items = ['Garlic Naan']   

    # Run ML Models and query single item - chosen for the purpose of demonstration. 
    # ML model will generate - Revenue, historical meat prices and a prediction model + bar chart of the included_items list.
    ml.sql_engine(include_items)

   
    # Send links of files to index.html - hard coded limited links to visualisation for demonstration purposes.
    return render_template('index.html',  
                            py_trends=f'static/images/SearchTrend-{nowDate}.png',                              
                            beef_pred=f'static/images/beef_pred-price.png',
                            chicken_pred=f'static/images/chicken_pred-price.png',
                            rev_pred_1=f'static/images/Revenue_pred-total($).png',
                            item_pred1=f'static/images/Garlic_Naan_pred-sold.png',
                            bar1=f"static/images/Garlic_Naan_bar.png"
                            )

    
if __name__ == '__main__':
    app.run(debug=True)