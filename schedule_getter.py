import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    home = input()
    away = input()
    link = "https://www.cbssports.com/college-basketball/teams/"
    f = requests.get(link)

    home_schedule = []
    away_schedule = []

    soup = BeautifulSoup(f.text, 'html.parser')
    last = ""
    for x in soup.body.find_all("span", class_="TeamName"):
        if x.string == home:
            print(x)