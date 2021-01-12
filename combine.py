from schedule_getter import schedule
from data_getter import data
from PyBrain import SupervisedDataSet

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
            homeStats = []
            awayStats = []
            for team in game:
                if i == 0:
                    homeStats= dat[team]
                elif i == 1:
                    awayStats = dat[team]
                else:
                    results.append(team)
                i = i + 1
            comb = []
            for x in range(len(homeStats)):
                comb.append(float(homeStats[x]) - float(awayStats[x]))
            games.append(comb)
    ds = SupervisedDataSet(len(games[0]), 1)
    for x in range(len(games)):
        ds.addSample(games[x], results[x])
    print(ds)
            
