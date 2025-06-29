# Web Scraping of ufcstats.com
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import datetime

# FIGHTERS
def fighter_scraper():
    data = []
    base_url = "http://ufcstats.com/statistics/fighters"
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    for letter in letters:
        url = f"{base_url}?char={letter}&page=all"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        fighters = soup.find_all('tr', class_='b-statistics__table-row')

        fighters.pop(0)
        fighters.pop(0)

        for fighter in fighters:
            def get_li(li_tag): # the information are inside the last content of the li tags
                contents = li_tag.contents
                return contents[-1].strip()
            def get_record(record):
                record = record.replace("Record: ", "") # we remove the initial text
                match = re.search(r'\((\d+)\s*NC\)', record) # we save the number in (# NC)
                nc = int(match.group(1)) if match else 0
                record = re.sub(r'\s*\(\d+\s*NC\)', '', record) # we remove the (# NC) part, we only want w-l-d
                parts = record.split('-')
                return parts[0], parts[1], parts[2], nc
            def get_height(height):
                parts = height.split("\' ")
                if(parts[0]=='--'):
                    return np.nan
                parts[1] = re.sub(r'"', '', parts[1])
                inches = int(parts[0])*12 + int(parts[1])
                return inches * 2.54
            def get_weight(weight):
                if (weight=='--'): 
                    return np.nan
                weight = re.sub(r'\s*lbs.', '', weight)
                return float(weight) * 0.4536
            def get_reach(reach):
                if (reach=='--'):
                    return np.nan
                reach = re.sub(r'"', '', reach)
                return int(reach)*2.54
            def get_date(date):
                if(date=='--'):
                    return ''
                date = datetime.strptime(date, "%b %d, %Y")
                date = date.strftime("%Y-%m-%d")
                return date
            

            link = fighter.find_all('a')[0]['href']
            print(link)
            response = requests.get(link)
            fighter_info= BeautifulSoup(response.content, 'html.parser')
            name = fighter_info.find('span', class_='b-content__title-highlight').text.strip()
            record = fighter_info.find('span', class_='b-content__title-record').text.strip()
            win, lost, draw, nc = get_record(record)
            li_items = fighter_info.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
            
            height = get_height(get_li(li_items[0]))
            weight = get_weight(get_li(li_items[1]))
            reach = get_reach(get_li(li_items[2]))
            stance = get_li(li_items[3])
            dob = get_date(get_li(li_items[4]))
            data.append([name, int(win), int(lost), int(draw), nc, height, weight, reach, stance, dob, link])

    df_fighters = pd.DataFrame(data, columns=["fighter_name", "fighter_win", "fighter_lost", "fighter_draw", "fighter_nc", "fighter_height", "fighter_weight", "fighter_reach", "fighter_stance", "fighter_DoB", "fighter_link"])
    return df_fighters
