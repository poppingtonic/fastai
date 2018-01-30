from glob2 import glob
import pandas as pd

df = DataFrame(columns=["file", "species"])

for image in glob("train/**/*.png"):
    dir_ = image.split('/')
    file_, species = dir_[-1], dir_[-2]

    df = df.append({"file": file_,
                    "species": species},
                   ignore_index=True)
df.to_csv('labels.csv', index=False)
