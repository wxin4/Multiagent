NOTE - (Use python2 or python whichever runs python 2 for your environment)
The setup is very similar to Project 2 ... Only additional thing is we have used matplotlib



# Setup and Initialization
* Operating System used windows.
* Install `python 2.7` and ensure it is in path
* Install matplotlib using -> python2 -m pip install matplotlib. (If matplot already installed and you are facing some issues then uninstall using pip uninstall matplotlib) followed by the above command to install matplotlib. 
* Clone the repository
* Go to `multiagent` directory
* Inside the directory run command `python pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5 -q`  
* If the above command runs fine, the setup is correct

# Multiagent

Commands that can be used for running :-

REMINDER - (Use python2 or python whichever runs python 2 for your environment)
In below commands I have used python2 ... If that doesn't work then try python ... If that also doesn't then use py -2


------------   minimaxClassic Layout with 2 ghosts ----------------------------

python pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5 -q

python2 pacman.py -p ExpectimaxAgent -l minimaxClassic -k 2 -n 5 -q

python2 pacman.py -p MinimaxAgent -l minimaxClassic -k 2 -n 5

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


