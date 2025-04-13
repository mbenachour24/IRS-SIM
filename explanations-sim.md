
📘 IRS-sim.py – Applied IRS framework, example on a 'consultancy firm'
⸻

🧩 Agent Roles

Each agent plays a distinct operational role in the organization:

Agent	Description
Consultants	Generate daily decisions (with a probability of being valid). They are the firm’s primary revenue drivers.
Experts	Validate consultant decisions. If valid, decisions generate revenue ($1000–$2000).
Middle Office	Processes validated decisions (administrative role).
Human Resources	Manages hiring and firing based on financial trends and consultant performance.
Security	Occasionally triggers events like audits or compliance checks, introducing random costs.
Executive Office	Adds consistent daily executive-level salary costs (CEO, CTO, CFO).
Accountants	Centralized unit to calculate daily revenue, cost, profit, and compute profit trends.
Investors	Receive dividends from profits based on their equity share.



⸻

🔢 IRS Model Calculations

The simulation incorporates a lightweight IRS model using three equations:
	•	Coupling Function:
C(f, g) = \frac{\text{Valid Revenue Decisions}}{f(\text{Output})}
	•	Efficiency:
K = \frac{\text{Valid Revenue Decisions}}{g(\text{Output})}
	•	System State Update:
S(t+1) = \frac{\text{Cash Balance}}{30} + K \cdot C(f, g)

These are logged over time and used to track organizational health.

⸻

💼 Key Simulation Features
	•	Performance-Driven HR:
	•	Hiring every 30 days based on profit and hiring strategy (aggressive, default, conservative).
	•	Firing every 100 days based on weighted scoring of consultant performance (revenue and validity).
	•	Random Events:
	•	Triggered randomly (~2% daily chance).
	•	Examples: Market downturns, new regulations, client lawsuits.
	•	Direct impact on cash balance.
	•	Profit Trend Analysis:
	•	Simple linear regression over the last 30 days of profit.
	•	Used by HR to dynamically adjust hiring strategy.
	•	Investor Dividends:
	•	Investors (each owning 20%) receive proportional payouts from daily profit.

⸻

📊 Outputs

The simulation produces:
	•	CSV Logs:
	•	irs_trajectory.csv – Daily cash balance over time.
	•	consultant_log.csv – Decisions per consultant with validation and revenue details.
	•	bilan_100_global.csv – 100-day interval reports with consultant metrics.
	•	Visualizations:
	•	irs_trajectory_plot_day_X.png – Cash balance over time.
	•	consultant_revenue_day_X.png – Revenue per consultant.
	•	consultant_efficiency_day_X.png – Validity ratios per consultant.
	•	irs_components_day_X.png – Visualization of IRS model variables C(f,g), K, S(t).
	•	Markdown Reports:
	•	100day_summary_day_X.md – Summary reports every 100 days, including top/bottom consultants and financial health.

⸻

⚙️ Default Runtime Behavior
	•	Simulation runs for 30,000 days.
	•	Generates visual and tabular summaries every 50 or 100 days.
	•	Produces detailed logs and tracks every decision and financial outcome.

You can reduce the number of days in simulate_days() for quicker test runs.

⸻

🧪 Future Enhancements (Ideas)
	•	Plug-in modules for ML/NLP decision-making agents
	•	GUI dashboard for live visualization
	•	More nuanced random events and investor behavior
	•	Integration of external market models or client demand

