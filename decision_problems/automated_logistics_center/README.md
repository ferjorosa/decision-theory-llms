# Logistics Center Automation

## Description
This problem models the strategic decision of an e-commerce company evaluating whether to upgrade its logistics center using robotic automation. The company must choose between three options: 
* Implementing a conventional robotic system, 
* Implementing a more advanced AI-powered system with greater risk and potential reward
* Making no investment at all.

To reduce uncertainty associated with the advanced system, the company can optionally conduct a feasibility test on its critical components. The test provides noisy signals about the system's likely performance and affects the final investment decision.

The scenario is modeled using an influence diagram implemented in [PyAgrum](https://pyagrum.readthedocs.io/), and involves sequential decision-making under uncertainty, expected utility analysis, and Bayesian updating based on test results.

## Author
Fernando Rodriguez Sanchez
