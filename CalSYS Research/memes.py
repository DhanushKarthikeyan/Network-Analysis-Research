

# full feature set from meme study -> basis of Ruicheng's paper


# Basic Network Features
'''
Number of Early Adopters: small number suggests few users generated most tweets

Size of First Surface: uninfected neighbors of early adopters

Size of Second Surface: uninfected neighbors of 2nd surface of early adopters

'''

# Distance Features -> position of adopters in the network
'''
Average Step Distance: avg shorted path length between consecutive users

CV of step distances: std over mean ... coefficent of variation - measures the relative variability in step distance

Diameter: max distance between two adopters
'''

# Community Features
'''
Number of Infected Communities: # of communities with at least one early adoper

Usage and Adopter Entropy: distribution of adopters across communities

Fraction of Intra-community user interaction: adopting from the same community / adopting from other community // lower found in early adopters of viral memes
'''

# Growth Rate Features -> time step duration is difference between tweets
'''
Average step time duration

CV of step time duration: std over mean
'''

