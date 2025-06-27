import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt



conn = sqlite3.connect(r"A:\Data Engineering\effilab_data_test.db")

query1 = """
SELECT 
c.id ,
c.activity_name ,
SUM(ds.spend) AS total_spend
FROM clients c 
JOIN daily_stats ds ON ds.client_id = c.id 
GROUP BY c.id 
ORDER BY total_spend DESC LIMIT 5
"""
query2 = """
SELECT 
STRFTIME('%Y-%m', ds.date) as month_,
SUM(ds.contacts) AS total_contacts
FROM daily_stats ds 
GROUP BY month_
"""
query3 = """
SELECT 
ds.client_id, 
STRFTIME('%Y-%m', ds.date) AS month_,
SUM(ds.spend) AS total_spend,
SUM(ds.contacts) AS total_contacts
FROM daily_stats ds 
GROUP BY ds.client_id , month_
"""
def run_query(query, conn): 
    print(query)
    query_ouput = pd.read_sql(query, conn)
    return query_ouput

df1 = run_query(query1, conn)
df2 = run_query(query2, conn)
df3 = run_query(query3, conn)


df1 = list(zip([]))

