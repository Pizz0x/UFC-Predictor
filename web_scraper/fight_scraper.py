# Web Scraping of ufcstats.com
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import datetime

def fight_scraper(url):
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

    def get_weightclass(weightclass):
        sex = 'F' if "Women" in weightclass else 'M'
        title_fight = "title" in weightclass
        weightclass = re.sub(r"\s*UFC\s*", "", weightclass)
        weightclass = re.sub(r"\s*Bout", "", weightclass)
        weightclass = re.sub(r"\s*Title", "", weightclass)
        return sex, title_fight, weightclass

    def get_result(result):
        result = re.sub(r'Method:\n\n\s*\n\s*', '', result)
        result = re.sub(r'-\s*', '', result)
        return result

    def get_round(round_format):
        match = re.search(r'\d+', round_format)
        return match.group()

    def get_succ_tot(f1, f2):
        part1 = f1.split(' of ')
        part2 = f2.split(' of ')
        return part1[0], part1[1], part2[0], part2[1]
    
    for event in events:
        event_link = event.find('a')['href']
        response = requests.get(event_link)
        soup = BeautifulSoup(response.content, 'html.parser')
        fights = soup.find_all('tr', class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
        c = 0
        for fight in fights:
            c += 1
            fight_link = fight['data-link']
            print(fight_link)
            f1_link = fight.find_all('a', class_="b-link b-link_style_black")[0]['href']
            f2_link = fight.find_all('a', class_="b-link b-link_style_black")[1]['href']
            fight_response = requests.get(fight_link)
            fight_info = BeautifulSoup(fight_response.content, 'html.parser')
            sex, title_fight, weightclass = get_weightclass(fight_info.find('i', class_='b-fight-details__fight-title').text.strip())
            if ('http://ufcstats.com/statistics/events/completed' in url):
                f1_outcome = fight_info.find_all('i', class_="b-fight-details__person-status")[0].text.strip()
                f2_outcome = fight_info.find_all('i', class_="b-fight-details__person-status")[1].text.strip()
                winner_link = '' if f1_outcome=='NC' else (f1_link if  f1_outcome=='W' else f2_link)
                result = get_result(fight_info.find('i', class_='b-fight-details__text-item_first').text.strip())
                i_items = fight_info.find_all('i', class_='b-fight-details__text-item')
                round = get_i(i_items[0])
                time = get_i(i_items[1])
                round_format = get_round(get_i(i_items[2]))
                stats_info = fight_info.find_all('td', 'b-fight-details__table-col')
                f1_knockdowns, f2_knockdowns = get_stat(stats_info[1])
                f1_sign_strikes_succ, f1_sign_strikes_tot, f2_sign_strikes_succ, f2_sign_strikes_tot = get_succ_tot(*get_stat(stats_info[2]))
                f1_total_strikes_succ, f1_total_strikes_tot, f2_total_strikes_succ, f2_total_strikes_tot = get_succ_tot(*get_stat(stats_info[4]))
                f1_takedowns_succ, f1_takedowns_tot, f2_takedowns_succ, f2_takedowns_tot = get_succ_tot(*get_stat(stats_info[5]))
                f1_sub, f2_sub = get_stat(stats_info[7])
                f1_ctrl, f2_ctrl = get_stat(stats_info[9])
                stats_data.append([fight_link, f1_link, int(f1_knockdowns), int(f1_sign_strikes_succ), int(f1_sign_strikes_tot), int(f1_total_strikes_succ), int(f1_total_strikes_tot), int(f1_takedowns_succ), int(f1_takedowns_tot), int(f1_sub), int(f1_ctrl)])
                stats_data.append([fight_link, f2_link, int(f2_knockdowns), int(f2_sign_strikes_succ), int(f2_sign_strikes_tot), int(f2_total_strikes_succ), int(f2_total_strikes_tot), int(f2_takedowns_succ), int(f2_takedowns_tot), int(f2_sub), int(f2_ctrl)])
            else:
                f1_outcome = f2_outcome = winner_link = result = round = time = ''
                round_format = 5 if (title_fight or c==1) else 3
            fight_data.append([fight_link, event_link, f1_link, f2_link, winner_link, f1_outcome, f2_outcome, sex, title_fight, weightclass, result, int(round), time, round_format])

    df_fights = pd.DataFrame(fight_data, columns=["fight_link", "event_link", "f1_link", 'f2_link','winner_link', 'f1_outcome', 'f2_outcome', 'sex', 'titlefight', 'weightclass', 'result', 'finish_round', 'time', 'round_format'])
    df_stats = pd.DataFrame(stats_data, columns=['fight_link', 'fighter_link', 'knockdowns', 'sign_strikes_succ', 'sign_strikes_att', 'total_strikes_succ', 'total_strikes_att', 'takedowns_succ', 'takedowns_att', 'sub', 'ctrl'])
    return df_fights, df_stats

    