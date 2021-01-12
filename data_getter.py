import requests
from bs4 import BeautifulSoup

def data():
    link = "https://www.sports-reference.com/cbb/seasons/2021-school-stats.html"
    f = requests.get(link)

    teams = {}
    soup = BeautifulSoup(f.text, 'html.parser')
    last = ""
    for y in soup.body.find_all("tr"):
        count = 0
        for x in y.find_all("td"):
            if x.string is not None:
                if x.get("data-stat") == "school_name":
                    teams[x.string] = []
                    last = x.string
                elif count > 2:
                    teams[last].append(x.string)
                else:
                    count = count + 1
    """
    for x in soup.body.find_all("td"):
        count = 0
        if x.string is not None:
            if x.get("data-stat") == "school_name":
                teams[x.string] = []
                last = x.string
            elif count > 4:
                teams[last].append(x.string)
            else:
                count = count + 1
    """
    
    link2 = "https://www.sports-reference.com/cbb/seasons/2021-opponent-stats.html"
    f2 = requests.get(link2)

    soup2 = BeautifulSoup(f2.text, 'html.parser')
    
    last = ""
    skip = 0
    for x in soup2.body.find_all("td"):
        if x.string is not None:
            if x.get("data-stat") == "school_name":
                last = x.string
                skip = 0
            elif skip > 14:
                teams[last].append(x.string)
            skip = skip + 1

    link3 = "https://www.sports-reference.com/cbb/seasons/2021-advanced-school-stats.html"
    f3 = requests.get(link3)

    soup3 = BeautifulSoup(f3.text, 'html.parser')
    
    last = ""
    skip = 0
    for x in soup3.body.find_all("td"):
        if x.string is not None:
            if x.get("data-stat") == "school_name":
                last = x.string
                skip = 0
            elif skip > 14:
                teams[last].append(x.string)
            skip = skip + 1

    link4 = "https://www.sports-reference.com/cbb/seasons/2021-advanced-opponent-stats.html"
    f4 = requests.get(link4)

    soup4 = BeautifulSoup(f4.text, 'html.parser')
    
    last = ""
    skip = 0
    for x in soup4.body.find_all("td"):
        if x.string is not None:
            if x.get("data-stat") == "school_name":
                last = x.string
                skip = 0
            elif skip > 15:
                teams[last].append(x.string)
            skip = skip + 1
    return teams

