print(f"\nTest cost tornado analysis completed.")
    tornado_data = tornado_results['tornado_data'][0]
    print(f"Total range of impact: {tornado_data['total_range']:.2f}")
    
    # Create and display the tornado plot
    fig_tornado = create_test_cost_tornado_plot(
        tornado_results=tornado_results,
        title="Oil Field Purchase: Test Cost Sensitivity Analysis"
    )
    
    # Create and display the sensitivity curve
    fig_curve = plot_test_cost_sensitivity_curve(
        tornado_results=tornado_results,
        title="Oil Field Purchase: Test Cost Sensitivity Curve"
    )
    
    print("\n" + "="*80)
    print("2. BREAKEVEN ANALYSIS")
    print("="*80)
    
    # Find breakeven point where decision changes
    breakeven_results = analyze_test_cost_breakeven(
        u_table=u_table,
        baseline_meu=baseline_meu,
        max_search_cost=200
    )
    
    print("\n" + "="*80)
    print("3. SPECIFIC TEST COST SCENARIOS")
    print("="*80)
    
    # Test specific cost scenarios
    test_scenarios = [0, 25, 50, 75, 100, 150]
    print(f"\nTesting specific cost scenarios: {test_scenarios}")
    
    scenario_results = []
    for cost in test_scenarios:
        u_modified = apply_test_cost_variation(u_table, cost)
        meu, decisions = update_utility_and_calculate_meu_with_decisions(u_modified)
        impact = meu - baseline_meu
        
        scenario_results.append({
            'cost': cost,
            'meu': meu,
            'impact': impact,
            'decisions': decisions
        })
        
        print(f"  Cost +{cost:3.0f}: MEU={meu:6.2f} (Impact: {impact:+6.2f}) | T={decisions['T']:6s} | B_policy={decisions['B']}")
    
    print("\n" + "="*80)
    print("4. DETAILED IMPACT ANALYSIS")
    print("="*80)
    
    print(f"\nSUMMARY OF TORNADO ANALYSIS:")
    print(f"Baseline MEU: {tornado_results['baseline_meu']:.2f}")
    print(f"Low impact scenario:  Cost {tornado_data['low_cost_variation']:+6.0f} → MEU {tornado_data['meu_low']:6.2f} (Impact: {tornado_data['low_impact']:+6.2f})")
    print(f"High impact scenario: Cost {tornado_data['high_cost_variation']:+6.0f} → MEU {tornado_data['meu_high']:6.2f} (Impact: {tornado_data['high_impact']:+6.2f})")
    print(f"Total sensitivity range: {tornado_data['total_range']:.2f}")
    print(f"Test cost affects {len([x for x in u_table['T'] if x == 'do'])} utility scenarios directly")
    
    # Show which utilities are affected
    print(f"\nUTILITIES AFFECTED BY TEST COST (T='do' cases):")
    do_cases = u_table[u_table['T'] == 'do']
    for idx, row in do_cases.iterrows():
        print(f"  {row['T']}, {row['B']}, {row['Q']}: {row['U']}")
    
    if breakeven_results:
        print(f"\nBREAKEVEN ANALYSIS:")
        print(f"Decision changes at test cost: +{breakeven_results['breakeven_cost']:.0f}")
        print(f"From '{breakeven_results['baseline_t_decision']}' to '{breakeven_results['breakeven_t_decision']}'")
        print(f"MEU at breakeven: {breakeven_results['breakeven_meu']:.2f}")
    else:
        print(f"\nBREAKEVEN ANALYSIS:")
        print(f"No decision change found within tested range")
    
    print("\n" + "="*80)
    print("5. VISUALIZATION")
    print("="*80)
    
    print("\nDisplaying tornado diagram...")
    fig_tornado.show()
    
    print("\nDisplaying sensitivity curve...")
    fig_curve.show()
    
    print("\n" + "="*80)
    print("6. EXTENDED ANALYSIS (±200 units)")
    print("="*80)
    
    # Extended analysis with larger range
    tornado_results_extended = analyze_test_cost_tornado(
        u_table=u_table,
        baseline_meu=baseline_meu,
        max_cost_variation=200,
        num_points=21
    )
    
    # Create extended plots
    fig_tornado_extended = create_test_cost_tornado_plot(
        tornado_results=tornado_results_extended,
        title="Oil Field Purchase: Extended Test Cost Sensitivity Analysis (±200)",
        height=400
    )
    
    fig_curve_extended = plot_test_cost_sensitivity_curve(
        tornado_results=tornado_results_extended,
        title="Oil Field Purchase: Extended Test Cost Sensitivity Curve (±200)"
    )
    
    tornado_data_extended = tornado_results_extended['tornado_data'][0]
    print(f"Extended analysis completed.")
    print(f"Total range of impact (±200): {tornado_data_extended['total_range']:.2f}")
    
    print("\nDisplaying extended tornado diagram...")
    fig_tornado_extended.show()
    
    print("\nDisplaying extended sensitivity curve...")
    fig_curve_extended.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nKEY INSIGHTS:")
    print(f"1. Test cost directly reduces utility for all 'do test' scenarios")
    print(f"2. Standard range (±100): Impact varies by {tornado_data['total_range']:.2f}")
    print(f"3. Extended range (±200): Impact varies by {tornado_data_extended['total_range']:.2f}")
    if breakeven_results:
        print(f"4. Decision changes at test cost: +{breakeven_results['breakeven_cost']:.0f}")
        print(f"5. Beyond breakeven, optimal strategy becomes '{breakeven_results['breakeven_t_decision']}'")
    else:
        print(f"4. No decision change observed within tested cost ranges")
        print(f"5. 'Do test' remains optimal even with high test costs")
    
    print(f"\nTest cost sensitivity analysis provides insights into:")
    print(f"- How much we can afford to pay for testing")
    print(f"- When testing becomes uneconomical")
    print(f"- The linear relationship between test cost and expected utility")