import gradio as gr
import pyagrum as gum
import pyagrum.lib.image as gumimage
import tempfile
import numpy as np
import pandas as pd

from pyagrum import InfluenceDiagram

# For debugging:
# import os
# os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib"

# Default tables
r_cpt = pd.DataFrame({
    "R | Q": ["pass", "fail"],
    "high": [0.95, 0.05],
    "medium": [0.7, 0.3],
    "low": [0.15, 0.85]
})

q_cpt = pd.DataFrame({
    "Q": ["high", "medium", "low"],
    "Probability": [0.35, 0.45, 0.2]
})

u_table = pd.DataFrame(
    [
        ['do', 'buy', 'high', 1220],
        ['do', 'buy', 'medium', 600],
        ['do', 'buy', 'low', -30],
        ['do', 'not_buy', '-', 320],
        ['not_do', 'buy', 'high', 1250],
        ['not_do', 'buy', 'medium', 630],
        ['not_do', 'buy', 'low', 0],
        ['not_do', 'not_buy', '-', 350],
    ],
    columns=['T', 'B', 'Q', 'U']
)

def create_influence_diagram(q_cpt, r_cpt, u_table):
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

    # Q probabilities
    influence_diagram.cpt(Q)[:] = q_cpt.iloc[:, 1].to_list()

    # R probabilities
    for q in ["high", "medium", "low"]:
        pass_prob = r_cpt.loc[r_cpt["R | Q"] == "pass", q].values[0]
        fail_prob = r_cpt.loc[r_cpt["R | Q"] == "fail", q].values[0]

        influence_diagram.cpt(R)[{"Q": q, "T": "do"}] = [pass_prob, fail_prob, 0.0]
        influence_diagram.cpt(R)[{"Q": q, "T": "not_do"}] = [0.0, 0.0, 1.0]

    # U utility values
    influence_diagram.utility(U)[{"T": "do", "B": "buy"}] = np.array(
        [u_table.iloc[0, 3], u_table.iloc[1, 3], u_table.iloc[2, 3]])[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}] = np.array(
        [u_table.iloc[3, 3]] * 3)[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}] = np.array(
        [u_table.iloc[4, 3], u_table.iloc[5, 3], u_table.iloc[6, 3]])[:, np.newaxis]
    influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}] = np.array(
        [u_table.iloc[7, 3]] * 3)[:, np.newaxis]

    return influence_diagram

def generate_policy_summary(infdiag, inference_engine):
    summaries = []

    for var_id in infdiag.nodes():
        if not infdiag.isDecisionNode(var_id):
            continue

        var_name = infdiag.variable(var_id).name()
        instantiations, _ = inference_engine.optimalDecision(var_name).argmax()
        action_labels = infdiag.variable(var_id).labels()

        # Check if it's a simple (unconditional) decision
        if all(len(inst) == 1 and var_name in inst for inst in instantiations):
            decision_index = instantiations[0][var_name]
            decision_label = action_labels[decision_index]
            summaries.append(f"For the variable '{var_name}', the optimal decision is to take the action '{decision_label}'.")
        else:
            # Conditional decisions based only on non-decision parents
            all_parents = infdiag.parents(var_id)
            chance_parents = [
                pid for pid in all_parents if not infdiag.isDecisionNode(pid)
            ]
            chance_parent_names = [infdiag.variable(pid).name() for pid in chance_parents]

            summaries.append(f"For the variable '{var_name}', the optimal decisions are:")

            for inst in instantiations:
                decision_index = inst[var_name]
                decision_label = action_labels[decision_index]

                conditions = []
                for parent in chance_parent_names:
                    parent_id = infdiag.idFromName(parent)
                    parent_labels = infdiag.variable(parent_id).labels()
                    parent_value_index = inst[parent]
                    parent_label = parent_labels[parent_value_index]
                    conditions.append(f"{parent} = '{parent_label}'")

                condition_str = ", ".join(conditions)
                summaries.append(f"- When {condition_str}, take action '{decision_label}'.")

    return "\n".join(summaries)


def show_inference(q_cpt_input, r_cpt_input, u_table_input):
    infdiag = create_influence_diagram(q_cpt_input, r_cpt_input, u_table_input)
    ie = gum.ShaferShenoyLIMIDInference(infdiag)
    ie.makeInference()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = tmp.name

    gumimage.exportInference(infdiag, img_path, engine=ie)

    summary_text = generate_policy_summary(infdiag, ie)
    return img_path, summary_text

# Gradio UI
with gr.Blocks(title="Oil Field Purchase Decision Analysis") as demo:
    gr.HTML("""
    <h1>Oil Field Purchase Decision Analysis</h1>
    """) # Main title

    gr.HTML("""
    <p>This decision support tool helps an oil company analyze whether to perform a geological test on a potential oil field and subsequently whether to purchase it. The analysis considers the uncertain quality of the field, the imperfect nature of the test, and the company's valuation of different outcomes to recommend an optimal strategy.</p>
    """) # Brief summary

    with gr.Accordion("Detailed Problem Description", open=False):
        gr.HTML("""
        <p>An oil company is considering the <span style="color: red"><b>decision</b></span> (<span style="color: red"><b>B</b></span>) to buy an oil field. The oil field can have three quality levels (<span style="color: purple"><b>Q</b></span>): high, medium, and low. The company obviously does not know the "real" qaulity of the field beforehand, but it can provide an estimation (i.e., <span style="color: purple"><b>uncertainty</b></span> ) using historical data and imagery. <b>It is willing to pay a higher price for the field as its quality increases</b>.</p>

        <p>Before making the buy decision, the company needs to <span style="color: red"><b>decide</b></span> (<span style="color: red"><b>T</b></span>) if it wants to perform a geological test. This test will have a certain cost and its results (<span style="color: purple"><b>R</b></span>) will not be exact predictions about the quality of the field, but will provide a report on the porosity of the reservoir (high porosity generally indicates greater oil potential). The test will not be infallible, and thus contain a certain degree of <span style="color: purple"><b>uncertainty</b></span>. The test can have two possible outcomes:</p>

        <ul>
            <li><b>Pass:</b> The porosity of the reservoir rock is equal to or greater than 15%, indicating significant oil potential.</li>
            <li><b>Fail:</b> The porosity of the reservoir rock is less than 15%, indicating low oil potential.</li>
        </ul>
                    
        <table>
        <tr>
            <td>
            <img src="https://raw.githubusercontent.com/ferjorosa/decision-theory-llms/main/decision_problems/oil_field_purchase/images/rock_porosity.jpg" alt="Rock porosity examples" width="600">
            </td>
        </tr>
        <tr>
            <td align="center">
            <i><b>Figure 1.</b> Reservoir quality illustrated through porosity and permeability characteristics</i>
            </td>
        </tr>
        </table>


        <p>The chronological sequence of the decision process is as follows:</p>

        <ol>
            <li>The company decides whether or not to perform the geological test.</li>
            <li>If the test is performed, the results are observed.</li>
            <li>The company decides whether or not to buy the oil field.</li>
        </ol>

        <p>There is still residual uncertainty in the problem that affects utility: <b>What is the actual state of the oil field?</b></p>

        <p>In this example, it seems logical for the company to buy the oil field after obtaining a "pass" result, but this is not always the case. It will depend on its specific a priori beliefs about the quality of the land (for example, based on its historical data on oil fields with similar characteristics), the intrinsic uncertainty of the test (for example, the test may give a positive result but the field is not actually suitable, or vice versa) and how the company values the possible consequences.</p>
        """) # Long description inside accordion

    gr.HTML("""
    <hr>
    <p>Enter the prior probability distribution of oil field quality and the conditional probability distribution of the porosity test result.</p>
    """) # Subtitle for inputs

    with gr.Row():
        with gr.Column():
            gr.HTML("<h3>Prior Probability Distribution of Oil Field Quality (<span style='color: purple'><b>Q</b></span>)</h3>")
            with gr.Accordion("More Information", open=False): # Accordion Title simplified
                gr.HTML("""
                <p>These probabilities represent the company's belief on the oil field quality. We can imagine they were estimated based on historical information from the oil company's past exploration of similar oil fields. For example, the company could have a classification model that predicts the oil field quality by using <a href="https://www.satimagingcorp.com/applications/energy/exploration/oil-exploration/">satellite imagery and geographical location data</a>.</p>

                <table>
                <tr>
                    <td>
                    <img src="https://raw.githubusercontent.com/ferjorosa/decision-theory-llms/main/decision_problems/oil_field_purchase/images/oil_field_image.jpg" alt="Oil field image" width="400">
                    </td>
                    <td>
                    <img src="https://raw.githubusercontent.com/ferjorosa/decision-theory-llms/main/decision_problems/oil_field_purchase/images/oil_field_heatmap.jpg" alt="Oil field heatmap" width="400">
                    </td>
                </tr>
                <tr>
                    <td colspan="2" align="center">
                    <i><b>Figure 2.</b> Satellite imagery and heatmap of the oil field</i>
                    </td>
                </tr>
                </table>

                """)
            q_cpt_input = gr.Dataframe(
                value=q_cpt,
                headers=list(q_cpt.columns),
                col_count=(2, "fixed"),
                row_count=(3, "fixed"),
                datatype=["str", "number"]
                # label="Prior Probability Distribution of Oil Field Quality" # Removed label
            )

            gr.HTML("<h3>Conditional Probability Distribution of the Porosity Test Result (<span style='color: purple'><b>R</b></span>)</h3>") # New Main Title for R
            with gr.Accordion("More Information", open=False): # Accordion for R
                gr.HTML("""
                <p>The results of the porosity test are directly related to the actual quality of the oil field (Q).
                In a perfect scenario, the test would be highly accurate.
                However, real-world tests are not perfect. The table below introduces these measurement
                imperfections by showing the conditional probability of each test result given the
                true quality of the oil field.</p>
                """)
            r_cpt_input = gr.Dataframe(
                value=r_cpt,
                headers=list(r_cpt.columns),
                col_count=(4, "fixed"),
                row_count=(2, "fixed"),
                datatype=["str", "number", "number", "number"]
                # label="Conditional Probability Distribution of the Porosity Test Result" # Removed label
            )

            gr.HTML("<h3>Utility Table (<span style='color: red'><b>U</b></span>)</h3>") # New Main Title for U
            with gr.Accordion("More Information", open=False): # Accordion for U
                gr.HTML("""
                <p>The utility table quantifies the expected value or 'utility' for each possible outcome in the decision problem. It reflects the company's preferences and the financial (or other) consequences associated with different sequences of decisions (whether to conduct a test, whether to buy the field) and the underlying quality of the oil field.</p>

                <p>For example:</p>
                <ul>
                    <li>Performing a test typically incurs a cost, which would be factored into the utility values for scenarios where 'T' is 'do'.</li>
                    <li>Successfully buying a high-quality field ('B' is 'buy', 'Q' is 'high') should result in a high utility.</li>
                    <li>Buying a low-quality field might lead to a lower utility or even a net loss.</li>
                    <li>Choosing not to buy the field might result in a baseline utility, potentially representing saved investment or a small return from an alternative venture.</li>
                </ul>

                <p>The values in this table are crucial for the influence diagram to calculate the optimal decision strategy that aims to maximize the company's expected utility.</p>
                """)
            u_table_input = gr.Dataframe(
                value=u_table,
                headers=list(u_table.columns),
                col_count=(4, "fixed"),
                row_count=(8, "fixed"),
                datatype=["str", "str", "str", "number"]
                # label="Utility Table" # Removed label
            )

        with gr.Column():
            result_img = gr.Image(label="Influence Diagram")
            result_text = gr.Textbox(label="Optimal Decision Summary", lines=10)

            submit_btn = gr.Button("Calculate Optimal Decision", variant="primary") # Updated text and variant

    submit_btn.click(
        fn=show_inference,
        inputs=[q_cpt_input, r_cpt_input, u_table_input],
        outputs=[result_img, result_text]
    )

if __name__ == "__main__":
    print("Attempting to launch Gradio with share=True...") # New debug print
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
        print("Gradio launch command executed.") # New debug print
    except Exception as e:
        print(f"Error during Gradio launch: {e}") # Catch launch errors
        import traceback
        traceback.print_exc()
