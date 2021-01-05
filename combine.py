from schedule_getter import schedule
from data_getter import data

if __name__ == "__main__":
    home = "Michigan"
    away = "Kent State"

    sched = schedule(home, away)
    dat = data()
    results = []
    games = []
    for batch in sched:
        for game in batch:
            i = 0
            g = []
            for team in game:
                if i < 2:
                    g.append(dat[team])
                else:
                    results.append(team)
                i = i + 1
            games.append(g)
    print(games)
    print(results)