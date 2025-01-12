
# Read a txt file
def read_txt(path: str) -> str:
    myfile = open(path, "rt")
    data = myfile.read()
    myfile.close()
    return data
