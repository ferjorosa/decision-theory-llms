# Oil Field Purchase Problem

## Description
This problem explores the decision-making process of an oil company considering the purchase of an oil field under uncertainty. The company can conduct a geological test before making a purchase decision, balancing the cost of the test with the potential insights it provides. 

The problem is modeled using an influence diagram implemented in [PyAgrum](https://pyagrum.readthedocs.io/).

## Directory Contents

Key components of this project:

*   **`notebook.ipynb`**: Interactive Jupyter notebook with problem formulation and analysis using both an influence diagram and a decision tree.
*   **`gradio_app/`**: Gradio web application: ([ferjorosa/oil-field-purchase-decision](https://huggingface.co/spaces/ferjorosa/oil-field-purchase-decision))
    *   `app.py`: Main application script
    *   `Dockerfile`: Container configuration for local testing and private deployment
    *   `requirements.txt`: Python dependencies
    *   `packages.txt`: System dependencies
*   **`images/`**: Image assets for the notebook and web app
*   **`problem.yaml`**: Structured problem description for LLM evaluation

## Locally launch the Gradio app

```shell
docker build -t oil-field-purchase-decision .
```

```shell
docker run -p 7860:7860 oil-field-purchase-decision
```

## Author
Fernando Rodriguez Sanchez