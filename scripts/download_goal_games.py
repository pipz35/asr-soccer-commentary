from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="/scratch/izar/philip/soccernet_data")
mySNdl.password = "s0cc3rn3t"

games = [
    "england_epl/2015-2016/2016-03-20 - 19-00 Manchester City 0 - 1 Manchester United",
    "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool",
    "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool",
    "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool",
    "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea",
    "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United",
    "europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Real Madrid 1 - 0 Liverpool",
    "europe_uefa-champions-league/2014-2015/2014-12-10 - 22-45 Barcelona 3 - 1 Paris SG",
    "europe_uefa-champions-league/2014-2015/2015-02-17 - 22-45 Paris SG 1 - 1 Chelsea",
    "europe_uefa-champions-league/2014-2015/2015-02-24 - 22-45 Manchester City 1 - 2 Barcelona",
    "europe_uefa-champions-league/2014-2015/2015-04-15 - 21-45 Paris SG 1 - 3 Barcelona",
    "europe_uefa-champions-league/2014-2015/2015-04-22 - 21-45 Real Madrid 1 - 0 Atl. Madrid",
    "europe_uefa-champions-league/2014-2015/2015-05-06 - 21-45 Barcelona 3 - 0 Bayern Munich",
    "europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Manchester City 1 - 2 Juventus",
    "europe_uefa-champions-league/2015-2016/2015-11-04 - 22-45 Bayern Munich 5 - 1 Arsenal",
    "europe_uefa-champions-league/2015-2016/2015-11-25 - 22-45 Juventus 1 - 0 Manchester City",
    "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG",
    "italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli",
    "spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid",
]

for game in games:
    print(f"Downloading: {game}")
    mySNdl.downloadGame(
        game=game,
        files=["1_224p.mkv", "2_224p.mkv"],
    )

print("Done!")
