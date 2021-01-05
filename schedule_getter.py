import requests
from bs4 import BeautifulSoup
from googlesearch import search

def sched_get(team, name):
    link = "https://www.sports-reference.com" + team
    f = requests.get(link)
    soup = BeautifulSoup(f.text, 'html.parser')
    games = []
    for x in soup.body.find_all("a"):
        if x is not None and x.string == "Polls, Schedule & Results":
            link1 = "https://www.sports-reference.com" + x.get("href")
            f1 = requests.get(link1)
            soup1 = BeautifulSoup(f1.text, 'html.parser')
            for y in soup1.body.find_all("tr"):
                away = False
                opponent = ""
                game = []
                diff = 0
                foundTeam = True
                away = False
                for w in y.find_all("td"):
                    if w.string is not None and w.string == "@":
                        away = True
                    if w.get("data-stat") == "time_game":
                        diff = 0
                    if w.string is not None and w.get("data-stat") == "game_location":
                        if w.string == "@":
                            away = True
                    if w.string is not None and w.get("data-stat") == "pts":
                        diff = int(w.string)
                    if w.string is not None and w.get("data-stat") == "opp_pts" and foundTeam:
                        diff = diff - int(w.string)
                        if away:
                            diff = diff * -1
                        if away:
                            game.append(opponent)
                            game.append(name)
                        else:
                            game.append(name)
                            game.append(opponent)
                        game.append(diff)
                        if game not in games:
                            games.append(game)
                    if w is not None and w.get("data-stat") == "opp_name":
                        y = w.find_all("a")
                        if len(y) == 0:
                            foundTeam = False
                        else:
                            opponent = y[0].string
    return games

def schedule(home, away):
    link = "https://www.sports-reference.com/cbb/seasons/2021-school-stats.html"
    f = requests.get(link)

    """
    Get the schedule and results for each home and away team so far,
    and five other similar teams' schedule and results
    """
    games = []
    soup = BeautifulSoup(f.text, 'html.parser')

    for x in soup.body.find_all("td"):
        if x.get("data-stat") == "school_name" and x.string == home:
            y = x.find_all("a")
            link1 = "https://www.sports-reference.com" + y[0].get("href")
            f1 = requests.get(link1)
            soup1 = BeautifulSoup(f1.text, 'html.parser')
            found = False
            for z in soup1.body.find_all("a"):
                if not found and z.get("href") is not None and "conferences" in z.get("href"):
                    link2 = "https://www.sports-reference.com" + z.get("href")
                    found = True
                    f2 = requests.get(link2)
                    soup2 = BeautifulSoup(f2.text, 'html.parser')
                    num = 0
                    for line in soup2.body.find_all("td"):
                        if (line.get("data-stat") is not None and 
                            line.get("data-stat") == "school_name" and line.string != home and num < 5):
                            link3 = line.find_all("a")
                            games.append(sched_get(link3[0].get("href"), line.string))
                            num = num + 1
        if x.get("data-stat") == "school_name" and x.string == away:
            y = x.find_all("a")
            link1 = "https://www.sports-reference.com" + y[0].get("href")
            f1 = requests.get(link1)
            soup1 = BeautifulSoup(f1.text, 'html.parser')
            found = False
            for z in soup1.body.find_all("a"):
                if not found and z.get("href") is not None and "conferences" in z.get("href"):
                    link2 = "https://www.sports-reference.com" + z.get("href")
                    found = True
                    f2 = requests.get(link2)
                    soup2 = BeautifulSoup(f2.text, 'html.parser')
                    num = 0
                    for line in soup2.body.find_all("td"):
                        if (line.get("data-stat") is not None and 
                            line.get("data-stat") == "school_name" and line.string != home and num < 5):
                            link3 = line.find_all("a")
                            games.append(sched_get(link3[0].get("href"), line.string))
                            num = num + 1
    return games

            