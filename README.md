# About the Project
This is a research project (Coop, Fall '25) that dives into convergence analysis for gradient descent and Nesterov's Accelerated Gradient(NAG) on convex functions. Function minimization is an integral part of processes in fields such as Machine Learning and Quantum Mechanics. Most of the functions considered are too complex to minimize using high-order numerical methods, such as Newton’s method. We explore Gradient Descent (GD), a first-order method that converges slowly. We then focus on Nesterov’s Accelerated Gradient (NAG), a first-order method with much faster convergence than GD. 

# Built with
- Python
- LaTeX

# Getting Started
## Prerequisites
This project requires an environment in which to run Python scripts, such as VS Code or any other suitable platform. 
## Installation
1. Clone the repo:
   ```git
   git clone https://github.com/bliss-hums/Gradient-Descent-and-Nesterov-s-Accelerated-Gradient-
   ```
2. Navigate to the directory:
   ```git
   cd Gradient-Descent-and-Nesterov-s-Accelerated-Gradient-
   ```
3. Change git remote URL:
   ```git
   git remote set-url origin <new-url>
   git remote -v
   ```
4. Download required libraries if necessary:
   ```git
   python -m pip install numpy
   python -m pip install matplotlib
   ```
# Usage
After installation, run the Python scripts in sequence. Each one will output a graph with the convergence analysis of GD and NAG for a specific function. All results are explained in the Technical Report PDF.

# Contributions
If you have any suggestions to make this better or have an idea of interesting next steps I can take, please fork the repo and create a pull request. Thanks!

#  Acknowledgements
- [Research Supervisor: Prof. Diane Guignard](https://www.linkedin.com/in/diane-guignard-3ab77655/)
- [README Template](https://github.com/othneildrew/Best-README-Template)
