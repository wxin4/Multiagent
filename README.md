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
Kindly enter the Reasoning Level Depth upto which you want to explore


NOTES:-

1) TrappedClassic runs very fast, so allowing agent to explore more reasoning levels for each game - (1 to 20 levels)

2) MinimaxClassic runs very slow, so allowing agent to explore less reasoning levels - (1 to 6 levels) ... level 6 might take a little more time - so if you want to avoid that you can give 5 as max reasoning level

Final ANALYSIS:-

The analysis is displayed towrds the end of execution of all levels. 
It shows the CLASSIFICATION of levels into :
UNDERTHINKING levels
OPTIMAL level
OVERTHINKING levels


