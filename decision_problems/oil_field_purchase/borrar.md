---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[15], line 2
      1 # Generate tornado diagram with 10% variation
----> 2 results = generate_tornado_diagram(q_cpt, r_cpt, u_table, variation_percent=10)

Cell In[14], line 324, in generate_tornado_diagram(q_cpt, r_cpt, u_table, variation_percent, title, figsize)
    322 # Calculate baseline expected utility
    323 baseline_meu = inference_engine.MEU()
--> 324 print(f"Baseline Expected Utility: {baseline_meu:.2f}")
    326 variation_factor = variation_percent / 100
    327 tornado_data = []

TypeError: unsupported format string passed to dict.__format__

meu_result
{'mean': 751.0, 'variance': 205639.0}
special variables
function variables
'mean' =
751.0
'variance' =
205639.0
len() =
2

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[18], line 2
      1 # Generate tornado diagram with 10% variation
----> 2 results = generate_tornado_diagram(q_cpt, r_cpt, u_table, variation_percent=10)

Cell In[17], line 332, in generate_tornado_diagram(q_cpt, r_cpt, u_table, variation_percent, title, figsize)
    329 tornado_data = []
    331 # Analyze each variable type using dedicated functions
--> 332 tornado_data.extend(analyze_q_probabilities(q_cpt, r_cpt, u_table, baseline_meu, variation_factor, variation_percent))
    333 tornado_data.extend(analyze_r_probabilities(q_cpt, r_cpt, u_table, baseline_meu, variation_factor, variation_percent))
    334 tornado_data.extend(analyze_utility_values(q_cpt, r_cpt, u_table, baseline_meu, variation_factor, variation_percent))

Cell In[17], line 107, in analyze_q_probabilities(q_cpt, r_cpt, u_table, baseline_meu, variation_factor, variation_percent)
    104 meu_low = update_influence_diagram_parameters(q_low, r_cpt, u_table)
    105 meu_high = update_influence_diagram_parameters(q_high, r_cpt, u_table)
--> 107 impact_low = meu_low - baseline_meu
    108 impact_high = meu_high - baseline_meu
    109 total_range = abs(meu_high - meu_low)

TypeError: unsupported operand type(s) for -: 'dict' and 'float'