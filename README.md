# Setup and Initialization
   
* Install `python 2.7` and ensure it is in path
* Clone the repository
* Inside the directory run command `python pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5 -q`  
* If the above command runs fine, the setup is correct

# Multiagent

Commands that can be used for running :-

(Use python2 for python2 environment ... otherwise python should be fine)

------------   minimaxClassic Layout with 2 ghosts ----------------------------

python pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5 -q

python2 pacman.py -p ExpectimaxAgent -l minimaxClassic -k 2 -n 5 -q

python pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5

python2 pacman.py -p ExpectimaxAgent -l minimaxClassic -k 2 -n 5


---------------------------------------------------------------------------------

------------   trappedClassic Layout with 1 ghost ----------------------------

python2 pacman.py -p MinimaxAgent -l trappedClassic -k 1 -n 5 -q
 
python2 pacman.py -p ExpectimaxAgent -l trappedClassic -k 1 -n 5 -q

python2 pacman.py -p MinimaxAgent -l trappedClassic -k 1 -n 5
 
python2 pacman.py -p ExpectimaxAgent -l trappedClassic -k 1 -n 5

-------------------------------------------------------------------------------

STEP 2:
After you execute any of the above command.
Kindly enter the Reasoning Level Depth upto which you want to explore. 

For example - You give 6
Then our agent will run 'n' games for reasoning level 1, level 2, ..till... level 6
Then finally our ANALYZER agent will display ANALYSIS of pacman performance through each of these levels and categorize 1 to 6 levels in three different categories as UNDERTHINKING, OPTIMAL and OVERTHINKING.


NOTES:-

1) TrappedClassic runs very fast, so allowing agent to explore more reasoning levels for each game - (1 to 8 levels)

2) MinimaxClassic runs very slow, so allowing agent to explore less reasoning levels - (1 to 6 levels) ... level 6 might take a little more time - so if you want to avoid that you can give 5 as max reasoning level

# Final analysis:-

The analysis is displayed towrds the end of execution of all levels. 
It shows the CLASSIFICATION of levels into :
UNDERTHINKING levels
OPTIMAL level
OVERTHINKING levels


