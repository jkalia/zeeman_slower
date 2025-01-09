This code optimizes the positions and lengths of a given set of solenoids to 
match the ideal magnetic field for an increasing-field Zeeman slower (ZS). 

I first generate the ideal field for the ZS analytically. Then, the optimizer 
is given a user-chosen solenoid configuration and is asked to find the best 
current and lengths of the solenoid sections to match the ideal magnetic field 
profile.
