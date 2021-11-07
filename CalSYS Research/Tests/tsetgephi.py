
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import datetime as dt

with open('pickleX34.p', 'rb') as f:
    load = pk.load(f)
    print('retrieved!')

nx.write_gexf(load, "test.gexf")
