from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="/scratch/izar/philip/soccernet_data")
mySNdl.password = "s0cc3rn3t"

games = [
    "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
    "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City",
]

for game in games:
    print(f"Downloading: {game}")
    mySNdl.downloadGame(
        game=game,
        files=["1_224p.mkv", "2_224p.mkv"],
    )

print("Download complete!")
