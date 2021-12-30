import yaml


cfg = None

with open("../../training/cfg/task/CubeBot.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)


print(cfg["env"]["resetDist"])
