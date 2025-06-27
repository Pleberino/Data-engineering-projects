import glob 
import pandas as pd 
import numpy as np
import xml.etree.ElementTree as ET 
from datetime import datetime 
import requests
import sqlite3
from bs4 import BeautifulSoup
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(BASE_DIR, 'code_log.txt')
csv_file = os.path.join(BASE_DIR, 'Largest_banks_data.csv')
exchange_rates = os.path.join(BASE_DIR, 'exchange_rate.csv')
db_name = os.path.join(BASE_DIR, 'Banks.db')
url = "https://web.archive.org/web/20230908091635/https://en.wikipedia.org/wiki/List_of_largest_banks"
table_attributes = ["Name","MC_USD_Billion"]
table_name = "Largest_banks"


def log_progress(message): 
    timestamp_format = '%Y-%h-%d-%H:%M:%S' # Year-Monthname-Day-Hour-Minute-Second 
    now = datetime.now() # get current timestamp 
    timestamp = now.strftime(timestamp_format) 
    with open(log_file,"a") as f: 
        f.write(timestamp + ' : ' + message + '\n')

def extract(url):
    df = pd.DataFrame(columns=table_attributes)
    html_page = requests.get(url).text
    data = BeautifulSoup(html_page, "html.parser")
    tables = data.find_all("tbody") # Tables
    rows = tables[0].find_all("tr") # Rows 

    for row in rows:
        cells = row.find_all("td") # Cells
        if len(cells) != 0:
            
            bank_name = cells[1].find_all("a")[1].text
            market_cap = float(cells[2].get_text(strip=True)[:-1].replace(',', ''))

            data_dict = {"Name": bank_name,
                         "MC_USD_Billion": market_cap}
            df1 = pd.DataFrame(data_dict, index=[0])
            df = pd.concat([df,df1], ignore_index=True)
    return df

def transform(df):
    exchange_rates_df = pd.read_csv(exchange_rates)
    dict = exchange_rates_df.set_index("Currency").to_dict()["Rate"]
    df['MC_GBP_Billion'] = [np.round(x*dict['GBP'],2) for x in df['MC_USD_Billion']]
    df['MC_EUR_Billion'] = [np.round(x*dict['EUR'],2) for x in df['MC_USD_Billion']]
    df['MC_INR_Billion'] = [np.round(x*dict['INR'],2) for x in df['MC_USD_Billion']]
    
    return df

def load_to_csv(df, csv_file):
    df.to_csv(csv_file)

def load_to_db(df, sql_connection, table_name):
    df.to_sql(table_name, sql_connection, if_exists="replace", index=False)

def run_query(query, sql_connection):
    print(query)
    query_output = pd.read_sql(query, sql_connection)
    print(query_output)

query1 = f"SELECT * FROM {table_name}"
query2 = f"SELECT AVG(MC_GBP_Billion) FROM {table_name}"
query3 = f"SELECT Name FROM {table_name} LIMIT 5"


log_progress('Preliminaries complete. Initiating ETL process')
df = extract(url)

log_progress('Data extraction complete. Initiating Transformation process')
df = transform(df)

log_progress('Data transformation complete. Initiating loading process')
load_to_csv(df, csv_file)

log_progress('Data saved to CSV file')

sql_connection = sqlite3.connect(db_name)
log_progress('SQL Connection initiated.')

load_to_db(df, sql_connection, table_name)

log_progress('Data loaded to Database as table. Running the query')

run_query(query1, sql_connection)
run_query(query2, sql_connection)
run_query(query3, sql_connection)

log_progress('Process Complete.')

sql_connection.close()
log_progress('Server Connection closed')