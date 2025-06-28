# Web Scraping of ufcstats.com
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import datetime

url = "http://ufcstats.com/statistics/events/completed"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
events = soup.find_all('tr', class_="b-statistics__table-row") # we get the insights from an html table, row by row

events.pop(0)
events.pop(0)
fight_data = []
stats_data = []

def get_i(i_tag): # the information are inside the last content of the li tags
    contents = i_tag.contents
    return contents[-1].strip()

def get_stat(stat_info):
     contents = stat_info.contents
     return contents[1].text.strip(), contents[3].text.strip()
     
for event in events:
    event_link = event.find('a')['href']
    response = requests.get(event_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    fights = soup.find_all('tr', class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
    for fight in fights:
        fight_link = fight.find('a', class_='b-flag')['href']
        fight_response = requests.get(fight_link)
        fight_info = BeautifulSoup(fight_response.content, 'html.parser')
        f1_link = fight_info.find_all('a', class_="b-link b-fight-details__person-link")[0]['href']
        f2_link = fight_info.find_all('a', class_="b-link b-fight-details__person-link")[1]['href']
        f1_outcome = fight_info.find_all('i', class_="b-fight-details__person-status")[0].text.strip()
        f2_outcome = fight_info.find_all('i', class_="b-fight-details__person-status")[1].text.strip()
        weightclass = fight_info.find('i', class_='b-fight-details__fight-title').text.strip()
        result = fight_info.find('i', class_='b-fight-details__text-item_first').text.strip()
        i_items = fight_info.find_all('i', class_='b-fight-details__text-item')
        round = get_i(i_items[0])
        time = get_i(i_items[1])
        round_format = get_i(i_items[2])
        stats_info = fight_info.find_all('td', 'b-fight-details__table-col')
        f1_knockdowns, f2_knockdowns = get_stat(stats_info[1])
        f1_sign_strikes, f2_sign_strikes = get_stat(stats_info[2])
        f1_total_strikes, f2_total_strikes = get_stat(stats_info[4])
        f1_takedowns, f2_takedowns = get_stat(stats_info[5])
        f1_sub, f2_sub = get_stat(stats_info[7])
        f1_ctrl, f2_ctrl = get_stat(stats_info[9])

        fight_data.append([fight_link, event_link, f1_link, f2_link, f1_outcome, f2_outcome, weightclass, result, round, time, round_format])
        stats_data.append([fight_link, f1_link, f1_knockdowns, f1_sign_strikes, f1_total_strikes, f1_takedowns, f1_sub, f1_ctrl])
        stats_data.append([fight_link, f2_link, f2_knockdowns, f2_sign_strikes, f2_total_strikes, f2_takedowns, f2_sub, f2_ctrl])

fight_df = pd.DataFrame(fight_data, columns=["fight_link", "event_link", "f1_link", 'f2_link', 'f1_outcome', 'f2_outcome', 'weightclass', 'result', 'round', 'time', 'round_format'])
stats_df = pd.DataFrame(stats_data, columns=['fight_link', 'fighter_link', 'knockdowns', 'sign_strikes', 'total_strikes', 'takedowns', 'sub', 'ctrl'])

print(fight_df.head(20))
print(stats_df.head(40))

    