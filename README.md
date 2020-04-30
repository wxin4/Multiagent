# Multiagent

Commands that can be used for running :-

(Use python2 for python2 environment ... otherwise python should be fine)

python pacman.py -p MinimaxAgent -l minimaxClassic -k 1 -n 10

python2 pacman.py -p MinimaxAgent -l trappedClassic -k 2 -n 5

python2 pacman.py -p ExpectimaxAgent -l minimaxClassic -k 2 -n 5
 
python2 pacman.py -p ExpectimaxAgent -l trappedClassic -k 2 -n 5

STEP 2:
After you execute any of the above command.
Kindly enter the Reasoning Level Depth upto which you want to explore


NOTES:-

1) TrappedClassic runs very fast, so allowing agent to explore more reasoning levels for each game - (1 to 20 levels)

2) MinimaxClassic runs very slow, so allowing agent to explore less reasoning levels - (1 to 7 levels) ... even 7 takes a lot of time .. so better to explore enter at Max 6 when asked for input

Final ANALYSIS:-

The analysis is displayed towrds the end of execution of all levels. 
It shows the CLASSIFICATION of levels into :
UNDERTHINKING levels
OPTIMAL level
OVERTHINKING levels


