from dataset import *

if __name__ == "__main__":
    data = Dataset("../springfield/scripts/test*",
                   "../springfield/Shows.xlsx",
                   "/home/aida/Projects/GenderBias/data/")
    #data.read()
    data.get_roles(method="lal", lal_path="/home/aida/Projects/LAL-Parser-master/")
