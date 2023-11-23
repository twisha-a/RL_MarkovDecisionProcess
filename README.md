# MarkovDecisionProcess
### Running the Script
#### Command Line Structure: 
`  python mdp_script.py <input_file> [options]  `
  - Replace `<input_file>` with the path to your input file.
  - `[options]` can be replaced with any combination of the following flags:
  - `-df` or `--discount_factor`: Sets the discount factor (float). Default is 0.9.
  - `-min` or `--minimize`: Enables minimization. No value needed.
  - `-tol` or `--tolerance`: Sets the tolerance for exiting value iteration (float). Default is 0.001.
  - `-iter` or `--iterations`: Sets the iteration cutoff for value iteration (integer). Default is 100.
  - `-v` or `--verbose`: Enables verbose mode. No value needed.
#### **Command Line Example:** 
` python mdp.py "D:/D Drive/college classes/ai/lab/lab3/data/maze example.txt" -df 0.9 -min -tol 0.001 -iter 100 `

