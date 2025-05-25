# Oil Field Purchase Problem

## Description
This problem explores the decision-making process of an oil company considering the purchase of an oil field under uncertainty. The company can conduct a geological test before making a purchase decision, balancing the cost of the test with the potential insights it provides. 

The problem is modeled using an influence diagram implemented in [PyAgrum](https://pyagrum.readthedocs.io/).

## Directory Contents

Key components of this project:

*   **`notebook.ipynb`**: Interactive Jupyter notebook with problem formulation and analysis using both an influence diagram and a decision tree.
*   **`app/`**: Gradio web application for exploring the decision problem.
    *   `gradio_app.py`: Main application script
    *   `Dockerfile`: Container configuration for local testing and private deployment
    *   `requirements.txt`: Python dependencies
    *   `apt.txt`: System dependencies
*   **`images/`**: Image assets for the notebook and web app
*   **`problem.yaml`**: Structured problem description for LLM evaluation

## Author
Fernando Rodriguez Sanchez