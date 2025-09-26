import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Dummy Data ---
portfolio_risk_score = 72
expected_loss = 48.5
high_risk_component = "Cables (45%)"
costliest_subcomponent = "Export Cable Joints (£12m)"

data = {
    "Component": ["Turbine","Turbine","Turbine","Cable","Cable","Foundation","Substation","Substation"],
    "Subcomponent": ["Blade","Gearbox","Nacelle","Inter-array","Export","Jacket Node","Transformer","Switchgear"],
    "Gen": ["Gen-3","Gen-3","Gen-3","Gen-2","Gen-2","Gen-2","AC","AC"],
    "Depth (m)": [80,80,80,60,60,95,90,40],
    "PoF": [0.20,0.12,0.10,0.25,0.18,0.10,0.15,0.12],
    "CoS (£m)": [4,7,5,12,15,18,15,8],
    "Risk Score": [65,60,55,75,70,68,70,60]
}
df = pd.DataFrame(data)

scenarios = pd.DataFrame({
    "Scenario":["Cable Joint Failure","Blade Erosion","Gearbox Breakdown","Transformer Fault","Jacket Weld Fatigue"],
    "Component":["Cable","Turbine","Turbine","Substation","Foundation"],
    "Cost (£m)":[12,1.5,7,15,18],
    "Downtime (days)":[60,10,45,90,120],
    "Risk Contribution (%)":[30,8,15,20,27]
})

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Input Form", "KPI Summary", "Database", "Risk Charts", "Risk Score Engine", "Loss Scenarios", "Portfolio Risk View", "Monte Carlo Simulation"])

if page == "Input Form":
    st.title("Offshore Wind Farm Input Form")
    st.subheader("Enter farm-specific details for tailored risk assessment")

    # --- General Info ---
    st.markdown("### General Information")
    farm_name = st.text_input("Wind Farm Name", "North Sea Example")
    location = st.selectbox("Location", ["North Sea", "Celtic Sea", "Baltic", "US East Coast", "Asia-Pacific", "Other"])
    commissioning_year = st.number_input("Commissioning Year", min_value=2000, max_value=2050, value=2028)
    project_capacity = st.number_input("Total Project Capacity (MW)", min_value=10, max_value=5000, value=1500)
    num_turbines = st.number_input("Number of Turbines", min_value=1, max_value=500, value=100)
    turbine_rating = st.number_input("Turbine Rating (MW)", min_value=2.0, max_value=25.0, value=15.0, step=0.5)
    depth = st.number_input("Water Depth (m)", min_value=0, max_value=300, value=80)
    distance_to_shore = st.number_input("Distance to Shore (km)", min_value=0, max_value=300, value=100)
    cable_length = st.number_input("Export Cable Length (km)", min_value=0, max_value=500, value=120)

    # --- Turbine Details ---
    st.markdown("### Turbine Details")
    turbine_oem = st.selectbox("Turbine OEM", ["Siemens Gamesa", "Vestas", "GE", "MHI Vestas", "Other"])
    turbine_model = st.text_input("Turbine Model / Generation", "SG 15.0-222 DD")
    rotor_diameter = st.number_input("Rotor Diameter (m)", min_value=50, max_value=300, value=222)
    hub_height = st.number_input("Hub Height (m)", min_value=50, max_value=200, value=120)
    blade_type = st.selectbox("Blade Material/Coating", ["Standard Glass Fibre", "Carbon Fibre Hybrid", "Anti-Erosion Coating", "Other"])

    # --- Foundation ---
    st.markdown("### Foundation / Support Structure")
    foundation_type = st.selectbox("Foundation Type", ["Monopile", "Jacket", "Floating Semi-Sub", "Spar", "Gravity Base"])
    if foundation_type == "Jacket":
        jacket_nodes = st.number_input("Number of Jacket Nodes", min_value=4, max_value=200, value=40)
    design_life = st.number_input("Design Lifetime (years)", min_value=20, max_value=40, value=25)
    corrosion_protection = st.selectbox("Cathodic Protection", ["Anodes", "ICCP", "Hybrid", "None"])

    # --- Cables ---
    st.markdown("### Cables")
    cable_type = st.selectbox("Cable Type", ["AC Inter-array", "AC Export", "HVDC Export"])
    num_cables = st.number_input("Number of Export Cables", min_value=1, max_value=20, value=4)
    cable_manufacturer = st.selectbox("Cable Manufacturer", ["NKT", "Prysmian", "LS Cable", "Other"])
    burial_depth = st.number_input(
    "Cable Burial Depth (m)", 
    min_value=0.0, 
    max_value=5.0, 
    value=1.5, 
    step=0.1
)

    joint_type = st.selectbox("Joint Type", ["Pre-moulded", "Cast Resin", "Shrink-fit"])

    # --- Substations ---
    st.markdown("### Substations")
    substation_type = st.selectbox("Substation Type", ["AC Offshore", "HVDC Offshore", "Onshore"])
    num_substations = st.number_input("Number of Substations", min_value=1, max_value=10, value=2)
    transformer_rating = st.number_input("Transformer Rating (MVA)", min_value=50, max_value=1000, value=800)
    switchgear_type = st.selectbox("Switchgear Type", ["AIS", "GIS"])
    if substation_type == "HVDC Offshore":
        converter_supplier = st.text_input("HVDC Converter Supplier", "Hitachi Energy")

    # --- O&M ---
    st.markdown("### Operations & Maintenance")
    om_strategy = st.radio("O&M Strategy", ["OEM Service", "In-house Operator", "Independent Service Provider"])
    warranty = st.checkbox("Major Component Warranty Active", value=True)
    vessel_strategy = st.selectbox("Vessel Strategy", ["Dedicated SOV", "CTV", "Spot-charter"])
    inspection_frequency = st.selectbox("Inspection Frequency", ["Annual", "Biennial", "Risk-based"])
    spare_parts = st.slider("Spare Parts Lead Time (days)", 0, 180, 30)

    # --- Environmental ---
    st.markdown("### Environmental & Site Conditions")
    metocean = st.slider("Metocean Downtime (% of year)", 0, 50, 15)
    seabed = st.selectbox("Seabed Type", ["Sand", "Clay", "Rock", "Mixed"])
    ice_risk = st.radio("Ice Risk", ["Yes", "No"])
    corrosion_category = st.selectbox("Corrosion Category (ISO 12944)", ["C4 High", "C5-M Marine", "CX Extreme"])
    extreme_wave = st.number_input("Extreme Wave Height Hs (50-year) (m)", min_value=0, max_value=30, value=15)

    # --- Submit ---
    if st.button("Generate Risk Profile"):
        st.success(f"Generating tailored risk profile for {farm_name}...")

        # Placeholder for risk model
        base_risk = 70
        if foundation_type == "Jacket":
            base_risk += 5
        if cable_length > 100:
            base_risk += 3
        if turbine_oem == "GE":  # Example OEM modifier
            base_risk += 2

        exp_loss = (num_turbines * turbine_rating * 0.05) + (cable_length * 0.1)

        # Display KPIs
        st.metric("Farm Risk Score", f"{base_risk}")
        st.metric("Expected Loss (£m)", f"{exp_loss:.1f}")
        st.metric("High-Risk Component", "Cables (Export)")
        st.metric("Costliest Subcomponent", "Transformer (£20m)")


if page == "KPI Summary":
    st.title("Offshore Wind Risk Management Framework")
    st.subheader("Prototype Dashboard for Insurance Underwriting")

    # --- Main KPIs (with captions for details) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio Risk Score", f"{portfolio_risk_score}")
        st.caption("Overall exposure rating")

    with col2:
        st.metric("Expected Loss", f"£{expected_loss}m")
        st.caption("Over 5-year projection")

    with col3:
        st.metric("High-Risk Component", "Cables")
        st.caption("45% of portfolio risk")

    with col4:
        st.metric("Costliest Subcomponent", "Transformer")
        st.caption("Estimated £20m replacement")

    # --- Secondary KPIs ---
    st.markdown("### Additional Portfolio Indicators")
    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric("Estimated Availability", "94.5 %")
        st.caption("Weighted by downtime risk")

    with col6:
        st.metric("O&M Cost Impact", "£22m")
        st.caption("Expected annualised OPEX")

    with col7:
        st.metric("Insurance Premium Index", "1.35x")
        st.caption("Relative to market baseline")

    # --- Loss Breakdown Donut ---
    st.markdown("### Loss Breakdown by Category")
    loss_breakdown = pd.DataFrame({
        "Category":["Repair Costs","Downtime Costs","Revenue Loss"],
        "Value":[20,15,13.5]
    })
    fig_donut = px.pie(loss_breakdown, names="Category", values="Value", hole=0.4,
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       title="Expected Loss Composition (£m)")
    st.plotly_chart(fig_donut, use_container_width=True)

    # --- Trend of Expected Loss over Years ---
    st.markdown("### Expected Loss Trend (Dummy Projection)")
    years = [2025,2026,2027,2028,2029]
    exp_loss = [40,42,45,47,48.5]
    fig_trend = px.line(x=years, y=exp_loss, markers=True, 
                        title="Projected Expected Loss (£m over Time)",
                        labels={"x":"Year","y":"Expected Loss (£m)"},
                        color_discrete_sequence=["tomato"])
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Interpretation ---
    st.markdown("""
    **Key Insights:**
    - The **portfolio risk score of 72** indicates a **Medium-High risk profile**, largely driven by cable exposure.  
    - The **expected financial loss (~£48.5m)** is dominated by **repair costs**, followed by downtime and lost revenue.  
    - **Availability is strong (94.5%)**, though insurance premiums are elevated at **1.35×** market baseline.  
    - Projections suggest a **gradual increase in losses** as farm scale and exposure grow.  
    """)


# --- Database ---
elif page == "Database":
    st.subheader("Component & Subcomponent Database")
    st.dataframe(df)

# --- Risk Charts ---
elif page == "Risk Charts":
    st.subheader("Risk Contribution by Component")
    comp_risk = {"Cables":45,"Turbines":25,"Foundations":20,"Substations":10}
    fig_pie = px.pie(names=comp_risk.keys(), values=comp_risk.values(), 
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Sunburst Chart ---
    st.subheader("Risk Breakdown by Component → Subcomponent")
    fig_sun = px.sunburst(df, path=["Component","Subcomponent"], values="Risk Score",
                          color="Risk Score", color_continuous_scale="YlOrRd")
    st.plotly_chart(fig_sun, use_container_width=True)

    # --- Bubble Chart ---
    st.subheader("Probability vs Consequence (Bubble Risk Map)")
    fig_bubble = px.scatter(df, x="PoF", y="CoS (£m)", size="Risk Score", color="Component",
                            hover_name="Subcomponent", 
                            labels={"PoF":"Probability of Failure","CoS (£m)":"Consequence (£m)"},
                            size_max=40, color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_bubble, use_container_width=True)

        # --- Risk by Technology Generation ---
    st.subheader("Risk Score by Technology Generation")
    gen_data = pd.DataFrame({
    "Gen":["Gen-2","Gen-2","Gen-2","Gen-2",
           "Gen-3","Gen-3","Gen-3","Gen-3"],
    "Component":["Turbine","Cable","Foundation","Substation",
                 "Turbine","Cable","Foundation","Substation"],
    "Risk Score":[60,70,55,65, 68,75,65,72],
    "Expected Loss (£m)":[7,12,8,10, 15,20,12,18]
    })

    # More realistic dataset
    gen_data = pd.DataFrame({
        "Gen":["Gen-2","Gen-2","Gen-2","Gen-2",
               "Gen-3","Gen-3","Gen-3","Gen-3"],
        "Component":["Turbine","Cable","Foundation","Substation",
                     "Turbine","Cable","Foundation","Substation"],
        "Risk Score":[60,70,55,65, 68,75,65,72],
        "Expected Loss (£m)":[7,12,8,10, 15,20,12,18]
    })

    # Bar for risk scores
    fig_gen = px.bar(gen_data, x="Gen", y="Risk Score", color="Component",
                     barmode="group", text="Risk Score",
                     color_discrete_sequence=px.colors.qualitative.Safe)

    # Line overlay for expected loss
    loss_summary = gen_data.groupby("Gen")["Expected Loss (£m)"].sum().reset_index()
    fig_gen.add_scatter(x=loss_summary["Gen"], y=loss_summary["Expected Loss (£m)"],
                        mode="lines+markers+text", name="Total Expected Loss (£m)",
                        line=dict(color="tomato", width=3, dash="dash"),
                        text=loss_summary["Expected Loss (£m)"],
                        textposition="top center")

    fig_gen.update_layout(
        title="Risk Score vs Expected Loss by Technology Generation",
        yaxis_title="Risk Score / Expected Loss (£m)",
        xaxis_title="Technology Generation",
        legend_title="Component",
        height=500
    )

    st.plotly_chart(fig_gen, use_container_width=True)



    # --- Risk Ranking Table ---
    st.subheader("Risk Ranking Table")
    styled_df = df[["Component","Subcomponent","PoF","CoS (£m)","Risk Score"]]\
                 .sort_values("Risk Score", ascending=False)
    st.dataframe(styled_df.style.background_gradient(cmap="Reds", subset=["Risk Score"]))

        # --- Tornado Sensitivity Chart ---
    st.subheader("Risk Sensitivity Analysis (Tornado Chart)")

    # Example sensitivity data (replace later with real model output)
    sensitivity_data = pd.DataFrame({
        "Parameter":["Probability of Failure (PoF)","Consequence Severity (CoS)","Metocean Downtime","Vessel Delay","Spare Part Lead Time"],
        "Impact":[25,20,15,10,5]  # dummy impact values
    }).sort_values("Impact", ascending=True)

    fig_tornado = go.Figure()

    fig_tornado.add_trace(go.Bar(
        x=sensitivity_data["Impact"],
        y=sensitivity_data["Parameter"],
        orientation="h",
        marker=dict(color="tomato"),
    ))

    fig_tornado.update_layout(
        title="Tornado Chart: Contribution of Parameters to Total Risk",
        xaxis_title="Impact on Risk Score",
        yaxis_title="Parameter",
        bargap=0.5,
        height=400
    )

    st.plotly_chart(fig_tornado, use_container_width=True)


# --- Risk Score Engine ---
elif page == "Risk Score Engine":
    st.subheader("Interactive Risk Score Engine")

    vessel_delay = st.slider("Vessel Delay (days)", 0, 30, 10)
    metocean = st.slider("Metocean Downtime (%)", 0, 40, 15)
    spare_parts = st.slider("Spare Parts Lead Time (days)", 0, 90, 20)

    base_score = 75
    adjusted = base_score + (vessel_delay*0.2) + (metocean*0.3) + (spare_parts*0.1)
    adjusted = min(100, adjusted)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adjusted,
        title={'text': "Risk Score"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "darkred" if adjusted>70 else "orange" if adjusted>40 else "green"},
               'steps': [
                   {'range': [0,40], 'color': "lightgreen"},
                   {'range': [40,70], 'color': "gold"},
                   {'range': [70,100], 'color': "tomato"}
               ]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Loss Scenarios ---
elif page == "Loss Scenarios":
    st.subheader("Loss Scenario Library")
    st.dataframe(scenarios)

    fig_loss = px.bar(scenarios, x="Scenario", y="Cost (£m)", color="Component",
                      text="Downtime (days)", color_discrete_sequence=px.colors.qualitative.Bold)
    fig_loss.update_traces(textposition="outside")
    st.plotly_chart(fig_loss, use_container_width=True)

# --- Portfolio Risk View ---
elif page == "Portfolio Risk View":
    st.subheader("Portfolio Risk Heatmap")
    heatmap_data = df.pivot(index="Component", columns="Subcomponent", values="Risk Score")
    fig_heatmap = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="YlOrRd")
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "Monte Carlo Simulation":
    st.subheader("Monte Carlo Risk Simulation")

    # --- User inputs ---
    st.markdown("### Define Uncertainty Ranges")
    vessel_min, vessel_max = st.slider("Vessel Delay Range (days)", 0, 60, (5, 20))
    metocean_min, metocean_max = st.slider("Metocean Downtime Range (%)", 0, 50, (10, 25))
    spare_min, spare_max = st.slider("Spare Parts Lead Time Range (days)", 0, 180, (15, 60))

    sims = st.number_input("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000)

    # --- Run simulation ---
    vessel_delay_samples = np.random.uniform(vessel_min, vessel_max, sims)
    metocean_samples = np.random.uniform(metocean_min, metocean_max, sims)
    spare_samples = np.random.uniform(spare_min, spare_max, sims)

    base_score = 75
    adjusted_scores = base_score + (vessel_delay_samples*0.2) + (metocean_samples*0.3) + (spare_samples*0.1)
    adjusted_scores = np.clip(adjusted_scores, 0, 100)

    # --- Convert to £ Financial Loss (dummy scaling model) ---
    # Assume CoS scaling: Risk Score (%) * max CoS (from df) / 100
    max_cost = df["CoS (£m)"].max()
    simulated_losses = (adjusted_scores/100) * max_cost * np.random.uniform(0.8, 1.2, sims)  # add noise factor

    # --- Summary statistics ---
    p10_risk, p50_risk, p90_risk = np.percentile(adjusted_scores, [10, 50, 90])
    p10_loss, p50_loss, p90_loss = np.percentile(simulated_losses, [10, 50, 90])

    st.markdown("### Simulation Results")
    st.write(f"**Risk Score (P10/P50/P90):** {p10_risk:.1f} / {p50_risk:.1f} / {p90_risk:.1f}")
    st.write(f"**Financial Loss (£m, P10/P50/P90):** {p10_loss:.2f} / {p50_loss:.2f} / {p90_loss:.2f}")

    # --- Parameter Summary Table ---
    param_data = {
        "Parameter":["Vessel Delay","Metocean Downtime","Spare Parts Lead Time"],
        "Min":[vessel_min, metocean_min, spare_min],
        "Max":[vessel_max, metocean_max, spare_max]
    }
    st.table(pd.DataFrame(param_data))

    # --- Risk Score Histogram ---
    fig_hist = px.histogram(adjusted_scores, nbins=40, title="Distribution of Risk Scores",
                            color_discrete_sequence=['tomato'])
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- CDF Plot ---
    sorted_scores = np.sort(adjusted_scores)
    cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
    fig_cdf = px.line(x=sorted_scores, y=cdf, title="Cumulative Distribution of Risk Scores")
    fig_cdf.update_layout(xaxis_title="Risk Score", yaxis_title="Cumulative Probability")
    st.plotly_chart(fig_cdf, use_container_width=True)

    # --- Financial Loss Histogram ---
    fig_loss = px.histogram(simulated_losses, nbins=40, 
                            title="Distribution of Simulated Financial Losses (£m)",
                            color_discrete_sequence=['royalblue'])
    st.plotly_chart(fig_loss, use_container_width=True)

    # --- Download Simulation Results ---
    sim_results = pd.DataFrame({
        "Risk Score": adjusted_scores,
        "Estimated Loss (£m)": simulated_losses
    })
    csv = sim_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Simulation Data as CSV", csv, "simulation_results.csv", "text/csv")


