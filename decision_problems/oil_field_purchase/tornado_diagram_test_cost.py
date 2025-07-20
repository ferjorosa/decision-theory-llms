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


def apply_test_cost_variation(u_table, test_cost_variation):
    """
    Apply test cost variation to utility table
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Original utility table
    test_cost_variation : float
        Amount to subtract from utilities when T='do' (positive means higher cost)
        
    Returns:
    --------
    pd.DataFrame : Modified utility table with test cost applied
    """
    u_modified = u_table.copy()
    u_modified = u_modified.astype({u_modified.columns[3]: float})
    
    # Apply test cost variation to all "do" cases
    # Indices 0, 1, 2, 3 correspond to 'do' cases in the utility table
    do_indices = u_modified[u_modified['T'] == 'do'].index
    u_modified.loc[do_indices, 'U'] = u_modified.loc[do_indices, 'U'] - test_cost_variation
    
    return u_modified


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


def analyze_test_cost_sensitivity(u_table, baseline_meu, cost_variations):
    """
    Analyze sensitivity of test cost for tornado diagram
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Original utility table
    baseline_meu : float
        Baseline maximum expected utility
    cost_variations : list
        List of test cost variations to analyze (e.g., [-50, -25, 0, 25, 50, 100])
        
    Returns:
    --------
    list : Tornado data for test cost variable
    """
    tornado_data = []
    
    print(f"\nAnalyzing Test Cost Sensitivity...")
    print(f"Cost variations to test: {cost_variations}")
    
    for cost_variation in cost_variations:
        # Apply test cost variation
        u_modified = apply_test_cost_variation(u_table, cost_variation)
        
        # Calculate MEU and optimal decisions
        meu, decisions = update_utility_and_calculate_meu_with_decisions(u_modified)
        
        impact = meu - baseline_meu
        
        tornado_data.append({
            'cost_variation': cost_variation,
            'meu': meu,
            'impact': impact,
            'decisions': decisions
        })
        
        print(f"  Test cost variation: {cost_variation:+.0f} → MEU: {meu:.2f} (Impact: {impact:+.2f})")
    
    return tornado_data


def analyze_test_cost_tornado(u_table, baseline_meu, cost_variation=50):
    """
    Analyze test cost sensitivity for tornado diagram with a single cost variation
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Original utility table
    baseline_meu : float
        Baseline maximum expected utility
    cost_variation : float
        Single cost variation value to analyze (will test both +cost_variation and -cost_variation)
        
    Returns:
    --------
    dict : Tornado analysis results for test cost
    """
    # Test both positive and negative variations
    cost_variations = [-cost_variation, cost_variation]
    
    print(f"Test Cost Tornado Analysis")
    print(f"Baseline MEU: {baseline_meu:.2f}")
    print(f"Cost variation: ±{cost_variation}")
    
    # Analyze sensitivity for both variations
    sensitivity_data = analyze_test_cost_sensitivity(u_table, baseline_meu, cost_variations)
    
    # Get low and high scenarios
    low_data = sensitivity_data[0]   # -cost_variation
    high_data = sensitivity_data[1]  # +cost_variation
    
    # Create tornado data in expected format
    tornado_data = [{
        'variable': 'Test Cost',
        'low_impact': low_data['impact'],
        'high_impact': high_data['impact'],
        'total_range': abs(high_data['impact'] - low_data['impact']),
        'low_cost_variation': low_data['cost_variation'],
        'high_cost_variation': high_data['cost_variation'],
        'meu_low': low_data['meu'],
        'meu_high': high_data['meu'],
        'decisions_low': low_data['decisions'],
        'decisions_high': high_data['decisions'],
        'cost_variation': cost_variation,
        'sensitivity_data': sensitivity_data
    }]
    
    return {
        'baseline_meu': baseline_meu,
        'tornado_data': tornado_data,
        'sensitivity_data': sensitivity_data,
        'cost_variation': cost_variation
    }


def create_test_cost_tornado_plot(tornado_results, title="Test Cost Tornado Diagram", 
                                 width=800, height=400):
    """
    Create tornado diagram for test cost sensitivity analysis
    
    Parameters:
    -----------
    tornado_results : dict
        Results from analyze_test_cost_tornado()
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
    tornado_data = tornado_results['tornado_data'][0]  # Only one variable
    baseline_meu = tornado_results['baseline_meu']
    cost_variation = tornado_results['cost_variation']
    
    # Extract data
    variable = tornado_data['variable']
    low_impact = tornado_data['low_impact']
    high_impact = tornado_data['high_impact']
    low_cost = tornado_data['low_cost_variation']
    high_cost = tornado_data['high_cost_variation']
    meu_low = tornado_data['meu_low']
    meu_high = tornado_data['meu_high']
    decisions_low = tornado_data['decisions_low']
    decisions_high = tornado_data['decisions_high']
    
    # Helper function to format decisions for hover text
    def format_decisions(decisions):
        t_decision = decisions['T']
        b_policy = decisions['B']
        b_text = ", ".join([f"{r}: {b}" for r, b in b_policy.items()])
        return f"T: {t_decision}, B: [{b_text}]"
    
    # Create the figure
    fig = go.Figure()
    
    # Determine which is left (negative) and right (positive)
    left_val = min(low_impact, high_impact)
    right_val = max(low_impact, high_impact)
    
    # Determine which scenario corresponds to left and right
    if low_impact < high_impact:
        left_meu = meu_low
        right_meu = meu_high
        left_decisions = decisions_low
        right_decisions = decisions_high
        left_cost = low_cost
        right_cost = high_cost
        left_label = f"Low Cost (Cost: {left_cost:+.0f})"
        right_label = f"High Cost (Cost: {right_cost:+.0f})"
    else:
        left_meu = meu_high
        right_meu = meu_low
        left_decisions = decisions_high
        right_decisions = decisions_low
        left_cost = high_cost
        right_cost = low_cost
        left_label = f"High Cost (Cost: {left_cost:+.0f})"
        right_label = f"Low Cost (Cost: {right_cost:+.0f})"
    
    # Create hover text
    left_hover_text = f"""
    Variable: {variable}<br>
    {left_label}<br>
    MEU: {left_meu:.2f}<br>
    Impact: {left_val:+.2f}<br>
    Optimal Decisions: {format_decisions(left_decisions)}
    """
    
    right_hover_text = f"""
    Variable: {variable}<br>
    {right_label}<br>
    MEU: {right_meu:.2f}<br>
    Impact: {right_val:+.2f}<br>
    Optimal Decisions: {format_decisions(right_decisions)}
    """
    
    # Add left bar
    fig.add_trace(go.Bar(
        y=[variable],
        x=[left_val],
        orientation='h',
        name='Lower Impact',
        marker=dict(color='lightcoral', line=dict(color='black', width=0.5)),
        offsetgroup=1,
        width=0.6,
        hoverinfo='skip'
    ))
    
    # Add right bar
    fig.add_trace(go.Bar(
        y=[variable],
        x=[right_val],
        orientation='h',
        name='Higher Impact',
        marker=dict(color='lightblue', line=dict(color='black', width=0.5)),
        offsetgroup=1,
        width=0.6,
        hoverinfo='skip'
    ))
    
    # Add invisible scatter points for hover
    fig.add_trace(go.Scatter(
        x=[left_val/2],
        y=[variable],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        hovertemplate=left_hover_text + '<extra></extra>',
        hoverlabel=dict(bgcolor='lightcoral', bordercolor='darkred', font=dict(color='black')),
        showlegend=False,
        name=''
    ))
    
    fig.add_trace(go.Scatter(
        x=[right_val/2],
        y=[variable],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        hovertemplate=right_hover_text + '<extra></extra>',
        hoverlabel=dict(bgcolor='lightblue', bordercolor='darkblue', font=dict(color='black')),
        showlegend=False,
        name=''
    ))
    
    # Add text annotations for impact values
    fig.add_annotation(
        x=left_val - (abs(left_val) * 0.1 if left_val != 0 else 1),
        y=variable,
        text=f'{left_val:.1f}',
        showarrow=False,
        font=dict(size=12),
        xanchor='right' if left_val < 0 else 'left'
    )
    
    fig.add_annotation(
        x=right_val + (abs(right_val) * 0.1 if right_val != 0 else 1),
        y=variable,
        text=f'{right_val:.1f}',
        showarrow=False,
        font=dict(size=12),
        xanchor='left' if right_val > 0 else 'right'
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line=dict(color='black', width=2))
    
    # Create subtitle
    subtitle = f'Baseline Expected Utility: {baseline_meu:.2f} | Test cost varied by ±{cost_variation} units'
    
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
            title='Variable',
            showticklabels=True
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


def plot_test_cost_sensitivity_curve(tornado_results, title="Test Cost Sensitivity Curve",
                                    width=800, height=500):
    """
    Create a sensitivity curve showing how MEU changes with test cost
    
    Parameters:
    -----------
    tornado_results : dict
        Results from analyze_test_cost_tornado()
    title : str
        Plot title
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure : The sensitivity curve figure
    """
    sensitivity_data = tornado_results['sensitivity_data']
    baseline_meu = tornado_results['baseline_meu']
    
    # Extract data for plotting
    cost_variations = [item['cost_variation'] for item in sensitivity_data]
    meus = [item['meu'] for item in sensitivity_data]
    impacts = [item['impact'] for item in sensitivity_data]
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add MEU curve
    fig.add_trace(go.Scatter(
        x=cost_variations,
        y=meus,
        mode='lines+markers',
        name='MEU',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        hovertemplate='Cost Variation: %{x:+.0f}<br>MEU: %{y:.2f}<extra></extra>'
    ))
    
    # Add baseline line
    fig.add_hline(
        y=baseline_meu,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f'Baseline MEU: {baseline_meu:.2f}',
        annotation_position="top right"
    )
    
    # Add vertical line at x=0 (no cost change)
    fig.add_vline(
        x=0,
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text='No Cost Change',
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Test Cost Variation (units)',
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title='Maximum Expected Utility',
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        width=width,
        height=height,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def analyze_test_cost_breakeven(u_table, baseline_meu, max_search_cost=200, tolerance=0.01):
    """
    Find the breakeven test cost where the optimal decision changes
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Original utility table
    baseline_meu : float
        Baseline maximum expected utility
    max_search_cost : float
        Maximum cost to search up to
    tolerance : float
        Tolerance for MEU comparison
        
    Returns:
    --------
    dict : Breakeven analysis results
    """
    print(f"\nFinding test cost breakeven point...")
    
    # Get baseline optimal decision
    _, baseline_decisions = update_utility_and_calculate_meu_with_decisions(u_table)
    baseline_t_decision = baseline_decisions['T']
    
    print(f"Baseline optimal T decision: {baseline_t_decision}")
    
    # Search for breakeven point
    cost_increment = 1.0
    current_cost = 0.0
    breakeven_cost = None
    
    while current_cost <= max_search_cost:
        u_modified = apply_test_cost_variation(u_table, current_cost)
        meu, decisions = update_utility_and_calculate_meu_with_decisions(u_modified)
        
        if decisions['T'] != baseline_t_decision:
            breakeven_cost = current_cost
            breakeven_meu = meu
            breakeven_decisions = decisions
            break
            
        current_cost += cost_increment
    
    if breakeven_cost is not None:
        print(f"Breakeven point found at test cost: +{breakeven_cost:.0f}")
        print(f"Optimal decision changes from '{baseline_t_decision}' to '{breakeven_decisions['T']}'")
        print(f"MEU at breakeven: {breakeven_meu:.2f}")
        
        return {
            'breakeven_cost': breakeven_cost,
            'baseline_t_decision': baseline_t_decision,
            'breakeven_t_decision': breakeven_decisions['T'],
            'breakeven_meu': breakeven_meu,
            'breakeven_decisions': breakeven_decisions
        }
    else:
        print(f"No breakeven point found within search range (0 to {max_search_cost})")
        return None


def analyze_single_test_cost_variation(u_table, baseline_meu, cost_variation):
    """
    Analyze the impact of a single test cost variation (both + and -)
    
    Parameters:
    -----------
    u_table : pd.DataFrame
        Original utility table
    baseline_meu : float
        Baseline maximum expected utility
    cost_variation : float
        Cost variation to analyze (will test both +cost_variation and -cost_variation)
        
    Returns:
    --------
    dict : Analysis results for the specific cost variation
    """
    print(f"\n{'='*60}")
    print(f"SINGLE TEST COST VARIATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Cost variation: ±{cost_variation}")
    print(f"Baseline MEU: {baseline_meu:.2f}")
    
    # Test negative variation (lower cost)
    u_lower = apply_test_cost_variation(u_table, -cost_variation)
    meu_lower, decisions_lower = update_utility_and_calculate_meu_with_decisions(u_lower)
    impact_lower = meu_lower - baseline_meu
    
    # Test positive variation (higher cost)
    u_higher = apply_test_cost_variation(u_table, cost_variation)
    meu_higher, decisions_higher = update_utility_and_calculate_meu_with_decisions(u_higher)
    impact_higher = meu_higher - baseline_meu
    
    print(f"\nRESULTS:")
    print(f"Lower cost (-{cost_variation}): MEU = {meu_lower:.2f}, Impact = {impact_lower:+.2f}")
    print(f"  Optimal decisions: T = {decisions_lower['T']}, B = {decisions_lower['B']}")
    print(f"Higher cost (+{cost_variation}): MEU = {meu_higher:.2f}, Impact = {impact_higher:+.2f}")
    print(f"  Optimal decisions: T = {decisions_higher['T']}, B = {decisions_higher['B']}")
    print(f"Total range: {abs(impact_higher - impact_lower):.2f}")
    
    return {
        'cost_variation': cost_variation,
        'baseline_meu': baseline_meu,
        'lower_cost': {
            'cost_variation': -cost_variation,
            'meu': meu_lower,
            'impact': impact_lower,
            'decisions': decisions_lower
        },
        'higher_cost': {
            'cost_variation': cost_variation,
            'meu': meu_higher,
            'impact': impact_higher,
            'decisions': decisions_higher
        },
        'total_range': abs(impact_higher - impact_lower)
    }


# Example usage and comprehensive analysis
if __name__ == "__main__":
    print("="*80)
    print("OIL FIELD PURCHASE: TEST COST SENSITIVITY ANALYSIS")
    print("="*80)
    
    print(f"\nBASELINE SCENARIO:")
    print(f"Baseline MEU: {baseline_meu:.2f}")
    print(f"Optimal T decision: {t_decision}")
    print(f"Optimal B policy: {b_policy}")
    
    # Show baseline utility table
    print(f"\nBASELINE UTILITY TABLE:")
    print(u_table.to_string(index=False))
    
    print("\n" + "="*80)
    print("1. SINGLE POINT ANALYSIS")
    print("="*80)
    
    # Example: Analyze impact of ±50 cost variation
    single_result = analyze_single_test_cost_variation(
        u_table=u_table,
        baseline_meu=baseline_meu,
        cost_variation=50
    )
    
    print("\n" + "="*80)
    print("2. TORNADO DIAGRAM ANALYSIS")
    print("="*80)
    
    # Analyze test cost sensitivity with tornado diagram for a specific value
    tornado_results = analyze_test_cost_tornado(
        u_table=u_table,
        baseline_meu=baseline_meu,
        cost_variation=50
    )
    
    # Create and show tornado plot
    tornado_fig = create_test_cost_tornado_plot(tornado_results)
    tornado_fig.show()
    
    print("\n" + "="*80)
    print("3. TRY DIFFERENT VALUES")
    print("="*80)
    
    # Example of trying different single values
    test_values = [25, 75, 100]
    for test_val in test_values:
        analyze_single_test_cost_variation(
            u_table=u_table,
            baseline_meu=baseline_meu,
            cost_variation=test_val
        )
