import gradio as gr
import pyagrum as gum
import pyagrum.lib.image as gumimage
import tempfile
import numpy as np
import pandas as pd
import itertools

from pyagrum import InfluenceDiagram

# Define R's default CPT as a DataFrame
qt_combinations = list(itertools.product(["high", "medium", "low"], ["do", "not_do"]))
default_cpt_R = [
    [0.95, 0.05, 0.0],  # Q=high, T=do
    [0.0, 0.0, 1.0],    # Q=high, T=not_do
    [0.7, 0.3, 0.0],    # Q=medium, T=do
    [0.0, 0.0, 1.0],    # Q=medium, T=not_do
    [0.15, 0.85, 0.0],  # Q=low, T=do
    [0.0, 0.0, 1.0],    # Q=low, T=not_do
]
default_r_cpt_df = pd.DataFrame(default_cpt_R, columns=["pass", "fail", "no_results"])
default_r_cpt_df.insert(0, "T", [t for q, t in qt_combinations])
default_r_cpt_df.insert(0, "Q", [q for q, t in qt_combinations])

def create_influence_diagram(q_high, q_medium, q_low, r_cpt_df):
    influence_diagram = InfluenceDiagram()

    Q = influence_diagram.addChanceNode(
        gum.LabelizedVariable("Q", "Q", 0).addLabel('high').addLabel('medium').addLabel('low'))
    R = influence_diagram.addChanceNode(
        gum.LabelizedVariable("R", "R", 0).addLabel('pass').addLabel('fail').addLabel('no_results'))
    T = influence_diagram.addDecisionNode(
        gum.LabelizedVariable("T", "T", 0).addLabel('do').addLabel('not_do'))
    B = influence_diagram.addDecisionNode(
        gum.LabelizedVariable("B", "B", 0).addLabel('buy').addLabel('not_buy'))
    U = influence_diagram.addUtilityNode(
        gum.LabelizedVariable("U", "U", 0).addLabel('utility'))

    influence_diagram.addArc("T", "R")
    influence_diagram.addArc("T", "B")
    influence_diagram.addArc("T", "U")
    influence_diagram.addArc("R", "B")
    influence_diagram.addArc("B", "U")
    influence_diagram.addArc("Q", "R")
    influence_diagram.addArc("Q", "U")

    # Normalize input probabilities for Q
    total = q_high + q_medium + q_low
    norm_q = [q_high / total, q_medium / total, q_low / total]
    influence_diagram.cpt(Q)[:] = norm_q

    # Set CPT for R based on DataFrame
    for _, row in r_cpt_df.iterrows():
        q_val = row["Q"]
        t_val = row["T"]
        probs = [row["pass"], row["fail"], row["no_results"]]
        influence_diagram.cpt(R)[{"Q": q_val, "T": t_val}] = probs

    # Utility values
    influence_diagram.utility(U)[{"T": "do", "B": "buy"}] = np.array([1250, 630, 0])[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}] = np.array([350, 350, 350])[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}] = np.array([1280, 660, 30])[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}] = np.array([380, 380, 380])[:, np.newaxis]

    return influence_diagram

def show_inference(q_high, q_medium, q_low, r_cpt_df):
    infdiag = create_influence_diagram(q_high, q_medium, q_low, r_cpt_df)
    ie = gum.ShaferShenoyLIMIDInference(infdiag)
    ie.makeInference()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = tmp.name

    gumimage.exportInference(infdiag, img_path, engine=ie)
    return img_path

# Gradio UI
with gr.Blocks(title="Oil Field Purchase Decision Analysis") as demo:
    gr.Markdown("""
    # Oil Field Purchase Decision Analysis  
    Enter probabilities for oil field quality and the CPT for R (they'll be used directly — no checks for normalization yet).
    """)

    with gr.Row():
        q_high = gr.Number(value=35, label="Q = high (%)")
        q_medium = gr.Number(value=45, label="Q = medium (%)")
        q_low = gr.Number(value=20, label="Q = low (%)")

    r_cpt_df_input = gr.Dataframe(
        value=default_r_cpt_df,
        headers=["Q", "T", "pass", "fail", "no_results"],
        col_count=(5, "fixed"),
        row_count=(6, "fixed"),
        label="Conditional Probability Table for R"
    )

    submit_btn = gr.Button("Submit")
    result_img = gr.Image(label="Influence Diagram with Policy")

    submit_btn.click(
        fn=show_inference,
        inputs=[q_high, q_medium, q_low, r_cpt_df_input],
        outputs=result_img
    )

if __name__ == "__main__":
    demo.launch()
