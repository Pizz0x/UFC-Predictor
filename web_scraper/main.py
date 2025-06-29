import os
from event_scaper import event_scraper
from fighter_scraper import fighter_scraper
from fight_scraper import fight_scraper

link_past = "http://ufcstats.com/statistics/events/completed?page=all"
link_next = "http://ufcstats.com/statistics/events/upcoming"

# # Get the Events
past_events = event_scraper(link_past)
link_next = event_scraper(link_next)

# # Get the Fighters
fighters = fighter_scraper()

# Get Fights and Stats
past_fights, past_stats = fight_scraper(link_past)
next_fights, next_stats = fight_scraper(link_next)

# now I have all the required dataframes: events, fighters, fights and stats for previous (training set) and next (prediction set) events
# what i want to do now is get the csv file from the dataframes:

# EVENTS
if os.path.exists('../data/past_events.csv'):
    os.remove('../data/past_events.csv')
past_events.to_csv('../data/past_events.csv', index=True, index_label='event_id', sep=',')
if os.path.exists('../data/next_events.csv'):
    os.remove('../data/next_events.csv')
past_events.to_csv('../data/next_events.csv', index=True, index_label='event_id', sep=',')

# FIGHTERS
if os.path.exists('../data/fighters.csv'):
    os.remove('../data/fighters.csv')
fighters.to_csv('../data/fighters.csv', index=True, index_label='fighter_id', sep=',')

# FIGHTS
if os.path.exists('../data/past_fights.csv'):
    os.remove('../data/past_fights.csv')
past_fights.to_csv('../data/past_fights.csv', index=True, index_label='fighter_id', sep=',')
if os.path.exists('../data/next_fights.csv'):
    os.remove('../data/next_fights.csv')
next_fights.to_csv('../data/next_fights.csv', index=True, index_label='fighter_id', sep=',')

# STATS
if os.path.exists('../data/past_stats.csv'):
    os.remove('../data/past_stats.csv')
past_stats.to_csv('../data/past_stats.csv', index=True, index_label='fighter_id', sep=',')
if os.path.exists('../data/next_stats.csv'):
    os.remove('../data/next_stats.csv')
next_stats.to_csv('../data/next_stats.csv', index=True, index_label='fighter_id', sep=',')


