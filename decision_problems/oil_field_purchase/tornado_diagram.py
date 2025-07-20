import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pyagrum as grum
import pyagrum.lib.notebook as gnb
from pyagrum import InfluenceDiagram

# Create influence diagram
influence_diagram = InfluenceDiagram()

Q = influence_diagram.addChanceNode(grum.LabelizedVariable("Q", "Q", 0).addLabel('high').addLabel('medium').addLabel('low'))
R = influence_diagram.addChanceNode(grum.LabelizedVariable("R", "R", 0).addLabel('pass').addLabel('fail').addLabel('no_results'))
T = influence_diagram.addDecisionNode(grum.LabelizedVariable("T", "T", 0).addLabel('do').addLabel('not_do'))
B = influence_diagram.addDecisionNode(grum.LabelizedVariable("B", "B", 0).addLabel('buy').addLabel('not_buy'))
U = influence_diagram.addUtilityNode(grum.LabelizedVariable("U", "U", 0).addLabel('utility'))

influence_diagram.addArc("T", "R")
influence_diagram.addArc("T", "B") # memory arc
influence_diagram.addArc("T", "U")
influence_diagram.addArc("R", "B")
influence_diagram.addArc("B", "U")
influence_diagram.addArc("Q", "R")
influence_diagram.addArc("Q", "U")

# Data setup
q_probs = [0.35, 0.45, 0.2]  # high, medium, low

r_cpt_data = {
    "high": [0.95, 0.05],   # [pass, fail]
    "medium": [0.7, 0.3],
    "low": [0.15, 0.85]
}

u_table = pd.DataFrame([
    ['do', 'buy', 'high', 1250],
    ['do', 'buy', 'medium', 630],
    ['do', 'buy', 'low', 0],
    ['do', 'not_buy', '-', 350],
    ['not_do', 'buy', 'high', 1280],
    ['not_do', 'buy', 'medium', 660],
    ['not_do', 'buy', 'low', 30],
    ['not_do', 'not_buy', '-', 380],
], columns=['T', 'B', 'Q', 'U'])

# Initialize influence diagram
# Q probabilities
influence_diagram.cpt(Q)[:] = q_probs

# R probabilities
for i, q_level in enumerate(["high", "medium", "low"]):
    pass_prob, fail_prob = r_cpt_data[q_level]
    influence_diagram.cpt(R)[{"Q": q_level, "T": "do"}] = [pass_prob, fail_prob, 0.0]
    influence_diagram.cpt(R)[{"Q": q_level, "T": "not_do"}] = [0.0, 0.0, 1.0]

# U utility values
influence_diagram.utility(U)[{"T": "do", "B": "buy"}] = np.array(
    [u_table.iloc[0, 3], u_table.iloc[1, 3], u_table.iloc[2, 3]])[:, np.newaxis]
influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}] = np.array(
    [u_table.iloc[3, 3]] * 3)[:, np.newaxis]
influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}] = np.array(
    [u_table.iloc[4, 3], u_table.iloc[5, 3], u_table.iloc[6, 3]])[:, np.newaxis]
influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}] = np.array(
    [u_table.iloc[7, 3]] * 3)[:, np.newaxis]

# Create inference engine
inference_engine = grum.ShaferShenoyLIMIDInference(influence_diagram)
inference_engine.makeInference()
meu_result = inference_engine.MEU()
baseline_meu = meu_result['mean']

# Get optimal decisions
optimal_T = inference_engine.optimalDecision("T").argmax()
optimal_B_tensor = inference_engine.optimalDecision("B")

# For T decision (simple decision)
t_decision = 'do' if optimal_T[0][0]['T'] == 0 else 'not_do'

# For B decision (depends on R), extract the policy
b_instantiations, _ = optimal_B_tensor.argmax()
b_policy = {}
for inst in b_instantiations:
    r_value = inst['R']
    b_value = inst['B']
    r_label = ['pass', 'fail', 'no_results'][r_value]
    b_label = 'buy' if b_value == 0 else 'not_buy'
    b_policy[r_label] = b_label

print(f"Optimal decision T: {t_decision}")
print(f"Optimal decision B: {b_policy}")
print(f"Baseline MEU: {baseline_meu:.2f}")


def update_utility_and_calculate_meu_with_decisions(u_table_modified):
    """
    Update utility values in the influence diagram and calculate MEU with optimal decisions
    
    Parameters:
    -----------
    u_table_modified : pd.DataFrame
        Modified utility table
        
    Returns:
    --------
    tuple : (MEU, optimal_decisions_dict)
    """
    # Store original utility values
    original_utilities = {}
    original_utilities["do_buy"] = influence_diagram.utility(U)[{"T": "do", "B": "buy"}].copy()
    original_utilities["do_not_buy"] = influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}].copy()
    original_utilities["not_do_buy"] = influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}].copy()
    original_utilities["not_do_not_buy"] = influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}].copy()
    
    try:
        # Update utility values
        influence_diagram.utility(U)[{"T": "do", "B": "buy"}] = np.array(
            [u_table_modified.iloc[0, 3], u_table_modified.iloc[1, 3], u_table_modified.iloc[2, 3]])[:, np.newaxis]
        influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}] = np.array(
            [u_table_modified.iloc[3, 3]] * 3)[:, np.newaxis]
        influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}] = np.array(
            [u_table_modified.iloc[4, 3], u_table_modified.iloc[5, 3], u_table_modified.iloc[6, 3]])[:, np.newaxis]
        influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}] = np.array(
            [u_table_modified.iloc[7, 3]] * 3)[:, np.newaxis]
        
        # Calculate MEU and optimal decisions
        temp_inference = grum.ShaferShenoyLIMIDInference(influence_diagram)
        temp_inference.makeInference()
        meu_result = temp_inference.MEU()
        meu = meu_result['mean']
        
        # Get optimal decisions
        optimal_T = temp_inference.optimalDecision("T").argmax()
        optimal_B_tensor = temp_inference.optimalDecision("B")
        
        # For T decision (simple decision)
        t_decision = 'do' if optimal_T[0][0]['T'] == 0 else 'not_do'
        
        # For B decision (depends on R), extract the policy
        b_instantiations, _ = optimal_B_tensor.argmax()
        b_policy = {}
        for inst in b_instantiations:
            r_value = inst['R']
            b_value = inst['B']
            r_label = ['pass', 'fail', 'no_results'][r_value]
            b_label = 'buy' if b_value == 0 else 'not_buy'
            b_policy[r_label] = b_label
        
        optimal_decisions = {
            'T': t_decision,
            'B': b_policy
        }
        
    finally:
        # Restore original values
        influence_diagram.utility(U)[{"T": "do", "B": "buy"}] = original_utilities["do_buy"]
        influence_diagram.utility(U)[{"T": "do", "B": "not_buy"}] = original_utilities["do_not_buy"]
        influence_diagram.utility(U)[{"T": "not_do", "B": "buy"}] = original_utilities["not_do_buy"]
        influence_diagram.utility(U)[{"T": "not_do", "B": "not_buy"}] = original_utilities["not_do_not_buy"]
    
    return meu, optimal_decisions


def update_utility_and_calculate_meu(u_table_modified):
    """
    Update utility values in the influence diagram and calculate MEU (backward compatibility)
    
    Parameters:
    -----------
    u_table_modified : pd.DataFrame
        Modified utility table
        
    Returns:
    --------
    float : Maximum Expected Utility
    """
    meu, _ = update_utility_and_calculate_meu_with_decisions(u_table_modified)
    return meu


def analyze_utility_sensitivity(u_table, baseline_meu, variation_amount=10, variation_type='percentage'):
    """
    Analyze sensitivity of utility values for tornado diagram
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Utility table
    baseline_meu : float
        Baseline maximum expected utility
    variation_amount : float
        Amount of variation to apply (percentage for 'percentage' type, absolute value for 'constant' type)
    variation_type : str
        Type of variation: 'percentage' or 'constant'
        
    Returns:
    --------
    list : Tornado data for utility variables
    """
    if variation_type == 'percentage':
        return analyze_utility_sensitivity_percentage(u_table, baseline_meu, variation_amount)
    elif variation_type == 'constant':
        return analyze_utility_sensitivity_constant(u_table, baseline_meu, variation_amount)
    else:
        raise ValueError("variation_type must be either 'percentage' or 'constant'")


def analyze_utility_sensitivity_percentage(u_table, baseline_meu, variation_percent=10):
    """
    Analyze sensitivity of utility values using percentage variations for tornado diagram
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Utility table
    baseline_meu : float
        Baseline maximum expected utility
    variation_percent : float
        Percentage variation to apply (e.g., 10 for ±10%)
        
    Returns:
    --------
    list : Tornado data for utility variables
    """
    tornado_data = []
    variation_factor = variation_percent / 100
    
    print(f"\nAnalyzing Utility Values (±{variation_percent}% variation)...")
    
    # Define utility scenarios to analyze
    utility_scenarios = [
        (0, 'U(do, buy, high)'),
        (1, 'U(do, buy, medium)'),
        (2, 'U(do, buy, low)'),
        (3, 'U(do, not_buy)'),
        (4, 'U(not_do, buy, high)'),
        (5, 'U(not_do, buy, medium)'),
        (6, 'U(not_do, buy, low)'),
        (7, 'U(not_do, not_buy)')
    ]
    
    for idx, scenario_name in utility_scenarios:
        # Create modified utility tables
        u_low = u_table.copy()
        u_high = u_table.copy()
        
        # Ensure the utility column is float type to avoid dtype warnings
        u_low = u_low.astype({u_low.columns[3]: float})
        u_high = u_high.astype({u_high.columns[3]: float})
        
        original_utility = u_table.iloc[idx, 3]
        
        # Apply percentage variation
        u_low.iloc[idx, 3] = original_utility * (1 - variation_factor)
        u_high.iloc[idx, 3] = original_utility * (1 + variation_factor)
        
        # Calculate expected utilities and optimal decisions
        meu_low, decisions_low = update_utility_and_calculate_meu_with_decisions(u_low)
        meu_high, decisions_high = update_utility_and_calculate_meu_with_decisions(u_high)
        
        impact_low = meu_low - baseline_meu
        impact_high = meu_high - baseline_meu
        total_range = abs(meu_high - meu_low)
        
        tornado_data.append({
            'variable': scenario_name,
            'low_impact': impact_low,
            'high_impact': impact_high,
            'total_range': total_range,
            'original_value': original_utility,
            'meu_low': meu_low,
            'meu_high': meu_high,
            'decisions_low': decisions_low,
            'decisions_high': decisions_high,
            'variation_type': 'percentage',
            'variation_amount': variation_percent
        })
        
        # Calculate the actual utility value changes
        low_utility_value = original_utility * (1 - variation_factor)
        high_utility_value = original_utility * (1 + variation_factor)
        low_change = low_utility_value - original_utility
        high_change = high_utility_value - original_utility
        
        print(f"  {scenario_name}: {original_utility:.0f} → Low={low_utility_value:.0f} ({low_change:+.0f}), High={high_utility_value:.0f} ({high_change:+.0f})")
    
    return tornado_data


def analyze_utility_sensitivity_constant(u_table, baseline_meu, variation_amount=50):
    """
    Analyze sensitivity of utility values using constant value variations for tornado diagram
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Utility table
    baseline_meu : float
        Baseline maximum expected utility
    variation_amount : float
        Constant amount to add/subtract (e.g., 50 for ±50 units)
        
    Returns:
    --------
    list : Tornado data for utility variables
    """
    tornado_data = []
    
    print(f"\nAnalyzing Utility Values (±{variation_amount} constant variation)...")
    
    # Define utility scenarios to analyze
    utility_scenarios = [
        (0, 'U(do, buy, high)'),
        (1, 'U(do, buy, medium)'),
        (2, 'U(do, buy, low)'),
        (3, 'U(do, not_buy)'),
        (4, 'U(not_do, buy, high)'),
        (5, 'U(not_do, buy, medium)'),
        (6, 'U(not_do, buy, low)'),
        (7, 'U(not_do, not_buy)')
    ]
    
    for idx, scenario_name in utility_scenarios:
        # Create modified utility tables
        u_low = u_table.copy()
        u_high = u_table.copy()
        
        # Ensure the utility column is float type to avoid dtype warnings
        u_low = u_low.astype({u_low.columns[3]: float})
        u_high = u_high.astype({u_high.columns[3]: float})
        
        original_utility = u_table.iloc[idx, 3]
        
        # Apply constant variation
        u_low.iloc[idx, 3] = original_utility - variation_amount
        u_high.iloc[idx, 3] = original_utility + variation_amount
        
        # Calculate expected utilities and optimal decisions
        meu_low, decisions_low = update_utility_and_calculate_meu_with_decisions(u_low)
        meu_high, decisions_high = update_utility_and_calculate_meu_with_decisions(u_high)
        
        impact_low = meu_low - baseline_meu
        impact_high = meu_high - baseline_meu
        total_range = abs(meu_high - meu_low)
        
        tornado_data.append({
            'variable': scenario_name,
            'low_impact': impact_low,
            'high_impact': impact_high,
            'total_range': total_range,
            'original_value': original_utility,
            'meu_low': meu_low,
            'meu_high': meu_high,
            'decisions_low': decisions_low,
            'decisions_high': decisions_high,
            'variation_type': 'constant',
            'variation_amount': variation_amount
        })
        
        # Calculate the actual utility value changes
        low_utility_value = original_utility - variation_amount
        high_utility_value = original_utility + variation_amount
        low_change = low_utility_value - original_utility
        high_change = high_utility_value - original_utility
        
        print(f"  {scenario_name}: {original_utility:.0f} → Low={low_utility_value:.0f} ({low_change:+.0f}), High={high_utility_value:.0f} ({high_change:+.0f})")
    
    return tornado_data


def create_utility_tornado_plot(tornado_data, baseline_meu, variation_amount=10, variation_type='percentage',
                               title="Utility Tornado Diagram", width=800, height=600):
    """
    Create tornado diagram using Plotly for utility sensitivity analysis
    
    Parameters:
    -----------
    tornado_data : list
        List of tornado data dictionaries
    baseline_meu : float
        Baseline expected utility
    variation_amount : float
        Amount of variation used (percentage or constant value)
    variation_type : str
        Type of variation: 'percentage' or 'constant'
    title : str
        Plot title
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure : The tornado plot figure
    """
    # Sort by total range (largest impact first)
    sorted_data = sorted(tornado_data, key=lambda x: x['total_range'], reverse=True)
    
    # Prepare data for plotting
    variables = [item['variable'] for item in sorted_data]
    low_impacts = [item['low_impact'] for item in sorted_data]
    high_impacts = [item['high_impact'] for item in sorted_data]
    original_values = [item['original_value'] for item in sorted_data]
    
    # Helper function to format decisions for hover text
    def format_decisions(decisions):
        t_decision = decisions['T']
        b_policy = decisions['B']
        b_text = ", ".join([f"{r}: {b}" for r, b in b_policy.items()])
        return f"T: {t_decision}, B: [{b_text}]"
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for low and high impacts
    for i, item in enumerate(sorted_data):
        var = item['variable']
        low = item['low_impact']
        high = item['high_impact']
        orig_val = item['original_value']
        meu_low = item['meu_low']
        meu_high = item['meu_high']
        decisions_low = item['decisions_low']
        decisions_high = item['decisions_high']
        
        # Determine which is left (negative) and right (positive)
        left_val = min(low, high)
        right_val = max(low, high)
        
        # Determine which scenario corresponds to left and right
        if low < high:
            left_meu = meu_low
            right_meu = meu_high
            left_decisions = decisions_low
            right_decisions = decisions_high
            left_label = "Low Scenario"
            right_label = "High Scenario"
        else:
            left_meu = meu_high
            right_meu = meu_low
            left_decisions = decisions_high
            right_decisions = decisions_low
            left_label = "High Scenario"
            right_label = "Low Scenario"
        
        # Create hover text for left bar
        left_hover_text = f"""
        Variable: {var}<br>
        {left_label}<br>
        MEU: {left_meu:.2f}<br>
        Impact: {left_val:+.2f}<br>
        Optimal Decisions: {format_decisions(left_decisions)}
        """
        
        # Create hover text for right bar  
        right_hover_text = f"""
        Variable: {var}<br>
        {right_label}<br>
        MEU: {right_meu:.2f}<br>
        Impact: {right_val:+.2f}<br>
        Optimal Decisions: {format_decisions(right_decisions)}
        """
        
        # Add left bar (from 0 to left_val) - no hover
        fig.add_trace(go.Bar(
            y=[var],
            x=[left_val],
            orientation='h',
            name='Lower Impact' if i == 0 else '',
            marker=dict(color='lightcoral', line=dict(color='black', width=0.5)),
            showlegend=i == 0,
            offsetgroup=1,
            width=0.6,
            hoverinfo='skip'
        ))
        
        # Add right bar (from 0 to right_val) - no hover
        fig.add_trace(go.Bar(
            y=[var],
            x=[right_val],
            orientation='h',
            name='Higher Impact' if i == 0 else '',
            marker=dict(color='lightblue', line=dict(color='black', width=0.5)),
            showlegend=i == 0,
            offsetgroup=1,
            width=0.6,
            hoverinfo='skip'
        ))
        
        # Add invisible scatter points for hover on left side
        fig.add_trace(go.Scatter(
            x=[left_val/2],
            y=[var],
            mode='markers',
            marker=dict(size=0.1, color='rgba(0,0,0,0)'),
            hovertemplate=left_hover_text + '<extra></extra>',
            hoverlabel=dict(bgcolor='lightblue', bordercolor='darkblue', font=dict(color='black')),
            showlegend=False,
            name=''
        ))
        
        # Add invisible scatter points for hover on right side
        fig.add_trace(go.Scatter(
            x=[right_val/2],
            y=[var],
            mode='markers',
            marker=dict(size=0.1, color='rgba(0,0,0,0)'),
            hovertemplate=right_hover_text + '<extra></extra>',
            hoverlabel=dict(bgcolor='lightcoral', bordercolor='darkred', font=dict(color='black')),
            showlegend=False,
            name=''
        ))
        
        # Add text annotations for impact values
        fig.add_annotation(
            x=left_val - (abs(left_val) * 0.1 if left_val != 0 else 1),
            y=var,
            text=f'{left_val:.1f}',
            showarrow=False,
            font=dict(size=10),
            xanchor='right' if left_val < 0 else 'left'
        )
        
        fig.add_annotation(
            x=right_val + (abs(right_val) * 0.1 if right_val != 0 else 1),
            y=var,
            text=f'{right_val:.1f}',
            showarrow=False,
            font=dict(size=10),
            xanchor='left' if right_val > 0 else 'right'
        )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line=dict(color='black', width=2))
    
    # Create subtitle based on variation type
    if variation_type == 'percentage':
        subtitle = f'Baseline Expected Utility: {baseline_meu:.2f} | Utility values varied by ±{variation_amount}%'
    else:  # constant
        subtitle = f'Baseline Expected Utility: {baseline_meu:.2f} | Utility values varied by ±{variation_amount} units'
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>{subtitle}</sub>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Change in Expected Utility from Baseline',
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            title='Utility Variables',
            categoryorder='array',
            categoryarray=variables[::-1]  # Reverse to show highest impact at top
        ),
        width=width,
        height=height,
        bargap=0.1,
        bargroupgap=0,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def generate_utility_tornado_data(u_table, baseline_meu, variation_amount=10, variation_type='percentage'):
    """
    Generate tornado analysis data for utility sensitivity analysis
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Utility table
    baseline_meu : float
        Baseline maximum expected utility
    variation_amount : float
        Amount of variation to apply (percentage for 'percentage' type, absolute value for 'constant' type)
    variation_type : str
        Type of variation: 'percentage' or 'constant'
        
    Returns:
    --------
    dict : Summary of results including baseline EU and variable impacts
    """
    print(f"Baseline Expected Utility: {baseline_meu:.2f}")
    
    # Analyze utility sensitivity
    tornado_data = analyze_utility_sensitivity(u_table, baseline_meu, variation_amount, variation_type)
    
    # Sort by total impact range (largest first)
    tornado_data.sort(key=lambda x: x['total_range'], reverse=True)
    
    return {
        'baseline_meu': baseline_meu,
        'tornado_data': tornado_data,
        'top_variable': tornado_data[0]['variable'] if tornado_data else None,
        'variation_amount': variation_amount,
        'variation_type': variation_type
    }


def plot_utility_tornado_diagram(tornado_results, title="Utility Tornado Diagram", 
                                width=800, height=600, show_plot=True):
    """
    Create and display tornado diagram from pre-computed tornado data
    
    Parameters:
    -----------
    tornado_results : dict
        Results from generate_utility_tornado_data()
    title : str
        Title for the plot
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
    show_plot : bool
        Whether to display the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure : The tornado plot figure
    """
    # Extract data from results
    tornado_data = tornado_results['tornado_data']
    baseline_meu = tornado_results['baseline_meu']
    variation_amount = tornado_results['variation_amount']
    variation_type = tornado_results['variation_type']
    
    # Create the tornado plot
    fig = create_utility_tornado_plot(tornado_data, baseline_meu, variation_amount, variation_type, title, width, height)
    
    if show_plot:
        fig.show()
    
    return fig


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PERCENTAGE VARIATION ANALYSIS")
    print("="*60)
    
    # Generate tornado analysis data with percentage variation
    tornado_results_percent = generate_utility_tornado_data(
        u_table=u_table,
        baseline_meu=baseline_meu,
        variation_amount=10,
        variation_type='percentage'
    )
    
    print(f"\nMost sensitive utility variable (% variation): {tornado_results_percent['top_variable']}")
    print(f"Total variables analyzed: {len(tornado_results_percent['tornado_data'])}")
    
    # Create and display the tornado plot for percentage
    fig_percent = plot_utility_tornado_diagram(
        tornado_results=tornado_results_percent,
        title="Oil Field Purchase: Utility Sensitivity Analysis (Percentage Variation)",
        show_plot=False
    )
    
    print("\n" + "="*60)
    print("CONSTANT VARIATION ANALYSIS")
    print("="*60)
    
    # Generate tornado analysis data with constant variation
    tornado_results_constant = generate_utility_tornado_data(
        u_table=u_table,
        baseline_meu=baseline_meu,
        variation_amount=100,  # ±100 units
        variation_type='constant'
    )
    
    print(f"\nMost sensitive utility variable (constant variation): {tornado_results_constant['top_variable']}")
    print(f"Total variables analyzed: {len(tornado_results_constant['tornado_data'])}")
    
    # Create and display the tornado plot for constant
    fig_constant = plot_utility_tornado_diagram(
        tornado_results=tornado_results_constant,
        title="Oil Field Purchase: Utility Sensitivity Analysis (Constant Variation)",
        show_plot=False
    )
    
    # Show both plots
    print("\nDisplaying percentage variation plot...")
    fig_percent.show()
    
    print("\nDisplaying constant variation plot...")
    fig_constant.show()


def compare_variation_methods(u_table, baseline_meu, percent_variation=10, constant_variation=100):
    """
    Compare percentage and constant variation methods side by side
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Utility table
    baseline_meu : float
        Baseline maximum expected utility
    percent_variation : float
        Percentage variation to use (e.g., 10 for ±10%)
    constant_variation : float
        Constant variation to use (e.g., 100 for ±100 units)
        
    Returns:
    --------
    tuple : (percentage_results, constant_results)
    """
    print("COMPARING VARIATION METHODS")
    print("="*50)
    
    # Percentage variation analysis
    print(f"1. Percentage Variation (±{percent_variation}%)")
    percent_results = generate_utility_tornado_data(
        u_table, baseline_meu, percent_variation, 'percentage'
    )
    
    print(f"\n2. Constant Variation (±{constant_variation} units)")
    constant_results = generate_utility_tornado_data(
        u_table, baseline_meu, constant_variation, 'constant'
    )
    
    # Compare top variables
    print(f"\nCOMPARISON SUMMARY:")
    print(f"Top variable (% method): {percent_results['top_variable']}")
    print(f"Top variable (constant method): {constant_results['top_variable']}")
    
    # Create comparison table
    print(f"\nTop 5 Most Sensitive Variables:")
    print(f"{'Rank':<4} {'Percentage Method':<25} {'Constant Method':<25}")
    print("-" * 54)
    
    percent_vars = [item['variable'] for item in percent_results['tornado_data'][:5]]
    constant_vars = [item['variable'] for item in constant_results['tornado_data'][:5]]
    
    for i in range(5):
        percent_var = percent_vars[i] if i < len(percent_vars) else "N/A"
        constant_var = constant_vars[i] if i < len(constant_vars) else "N/A"
        print(f"{i+1:<4} {percent_var:<25} {constant_var:<25}")
    
    return percent_results, constant_results