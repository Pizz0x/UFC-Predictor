# Web Scraping of ufcstats.com
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import datetime

# EVENTS
url = "http://ufcstats.com/statistics/events/completed?page=all"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
events = soup.find_all('tr', class_="b-statistics__table-row") # we get the insights from an html table, row by row

events.pop(0)
events.pop(0)
data = []

def get_location(location): # get city and country from location
    parts = location.split(', ')
    if (len(parts)==3):
        return parts[0], parts[2]
    else:
        return parts[0], parts[1]
def get_date(date):
    date = datetime.strptime(date, "%B %d, %Y")
    date = date.strftime("%Y-%m-%d")
    return date

    
for event in events: 
    name = event.find('a').text.strip()
    date = get_date(event.find('span').text.strip())
    location = event.find_all('td')[1].text.strip()
    city, country = get_location(location)
    link = event.find('a')['href']
    data.append([name, date, city, country, link]) # we save the useful data in a list, to then trasform it to a dataframe

df_event = pd.DataFrame(data, columns=["event_name", "event_date", "event_city", "event_country", "event_link"])

if os.path.exists('../data/df_events.csv'):
    os.remove('../data/df_events.csv')
df_event.to_csv('../data/df_events.csv', index=True, index_label='event_id', sep=',')

