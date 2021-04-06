from dataset import *

if __name__ == "__main__":
    data = Dataset("../springfield/scripts/test*", "../springfield/Shows.xlsx", "/home/aida/Projects/GenderBias/")
    #data.read()
    data.get_roles()
    #data.rearange()
