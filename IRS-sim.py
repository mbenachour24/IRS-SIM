import random
import names
from collections import defaultdict
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- Human Entity ---
class Human:
    def __init__(self, name, role, salary):
        self.name = name
        self.role = role
        self.salary = salary
        self.total_earnings = 0
        self.days_employed = 0

    def __repr__(self):
        return f"{self.name} ({self.role}) - ${self.salary}/month"

# --- Base IRS Agent Class ---
class IRSAgent:
    def __init__(self, name):
        self.name = name
        self.memory = []
        self.humans = []

    def add_humans(self, count, role, salary):
        for _ in range(count):
            self.humans.append(Human(name=names.get_full_name(), role=role, salary=salary))

    def act(self, state, context=None):
        pass

# --- Consultants ---
class Consultants(IRSAgent):
    def __init__(self):
        super().__init__("Consultants")
        self.add_humans(10, "Consultant", 10_000)
        self.files = []  # track decisions per consultant
        self.performance_history = defaultdict(list)  # Track performance over time

    def act(self, state, context=None):
        total_decisions = 0
        for human in self.humans:
            num_decisions = random.randint(1, 3)
            for _ in range(num_decisions):
                decision_valid = random.random() < 0.75  # % chance of being valid 
                self.files.append({
                    "consultant": human.name,
                    "decision": True,
                    "valid": decision_valid,
                    "reviewed": False,
                    "revenue": 0,
                    "day": state.get("day", 0) if state else 0
                })
                total_decisions += 1
        self.memory.append({"decisions_today": total_decisions})
        return self.files[-total_decisions:]

    def update_performance_history(self, day):
        # Calculate performance metrics for each consultant
        revenue_map = defaultdict(int)
        decisions_map = defaultdict(int)
        valid_decisions_map = defaultdict(int)
        
        # Only consider files from the last 30 days
        recent_files = [f for f in self.files if f.get("day", 0) > day - 30]
        
        for f in recent_files:
            consultant = f["consultant"]
            revenue_map[consultant] += f["revenue"]
            decisions_map[consultant] += 1
            if f["valid"]:
                valid_decisions_map[consultant] += 1
        
        # Update history for each consultant
        for human in self.humans:
            efficiency = valid_decisions_map[human.name] / max(1, decisions_map[human.name])
            revenue = revenue_map[human.name]
            
            self.performance_history[human.name].append({
                "day": day,
                "revenue": revenue,
                "decisions": decisions_map[human.name],
                "valid_ratio": efficiency,
                "revenue_per_decision": revenue / max(1, decisions_map[human.name])
            })
        
        return self.performance_history

# --- Experts ---
class Experts(IRSAgent):
    def __init__(self):
        super().__init__("Experts")
        self.add_humans(10, "Expert", 15_000)

    def act(self, state, context):
        files = context.get("files", [])
        validated_files = []
        for file in files:
            file["reviewed"] = True
            if file["valid"]:
                # Stochastic revenue: 5% no gain, 50% half gain
                roll = random.random()
                if roll < 0.05:
                    file["revenue"] = 0
                elif roll < 0.55:
                    file["revenue"] = 1000
                else:
                    file["revenue"] = 2000
                validated_files.append(file)
        self.memory.append(len(validated_files))
        return validated_files

# --- Middle Office ---
class MiddleOffice(IRSAgent):
    def __init__(self):
        super().__init__("Middle Office")
        self.add_humans(10, "Middle Office", 5_000)

    def act(self, state, context):
        processed = len(context.get("files", []))
        self.memory.append(processed)
        return processed

# --- HR ---
class HumanResources(IRSAgent):
    def __init__(self, consultants_agent):
        super().__init__("Human Resources")
        self.add_humans(10, "HR", 3_000)
        self.consultants = consultants_agent
        self.day_counter = 0
        self.hiring_strategy = "default"  # Can be 'default', 'aggressive', 'conservative'

    def act(self, state, context):
        self.day_counter += 1

        # Update hiring strategy based on profit trends
        if self.day_counter % 50 == 0:
            profit_trend = state.get("profit_trend", 0)
            if profit_trend > 0.1:  # Strong positive trend
                self.hiring_strategy = "aggressive"
            elif profit_trend < -0.1:  # Strong negative trend
                self.hiring_strategy = "conservative"
            else:
                self.hiring_strategy = "default"

        # Hire based on profit and strategy
        if self.day_counter % 30 == 0 and state["profit"] > 0:
            if self.hiring_strategy == "aggressive":
                new_hires = min(3, max(1, int(state["profit"] / 20000)))
                self.consultants.add_humans(new_hires, "Consultant", 10_000)
                print(f"📈 30-day profit window met — hired {new_hires} consultants (Aggressive Strategy).")
            elif self.hiring_strategy == "conservative":
                if state["profit"] > 30000:  # Higher threshold
                    self.consultants.add_humans(1, "Consultant", 10_000)
                    print(f"📈 30-day profit window met — hired 1 consultant (Conservative Strategy).")
            else:  # Default strategy
                self.consultants.add_humans(1, "Consultant", 10_000)
                print(f"📈 30-day profit window met — hired 1 consultant.")

        # Every 100 days: fire based on performance metrics
        if self.day_counter % 100 == 0 and len(self.consultants.humans) > 5:
            # Get performance data
            self.consultants.update_performance_history(self.day_counter)
            
            # Build comprehensive performance score
            performance_scores = []
            for consultant in self.consultants.humans:
                if consultant.days_employed < 90:  # Grace period
                    continue
                name = consultant.name
                history = self.consultants.performance_history.get(name, [])
                
                if len(self.consultants.humans) - len(to_fire) < 10:
                    to_fire = to_fire[:len(self.consultants.humans) - 10]

                if not history:
                    performance_scores.append((name, 0))
                    continue
                
                # Use recent history for scoring
                recent = history[-min(3, len(history)):]
                avg_revenue = sum(entry["revenue"] for entry in recent) / len(recent)
                avg_validity = sum(entry["valid_ratio"] for entry in recent) / len(recent)
                
                # Weighted score (revenue 70%, validity 30%)
                score = (0.7 * avg_revenue) + (0.3 * avg_validity * 2000)
                performance_scores.append((name, score))

            num_to_fire = max(1, int(len(performance_scores) * 0.10))  # Fire bottom 10% max
            worst = sorted(performance_scores, key=lambda x: x[1])[:num_to_fire]
            if state.get("profit_trend", 0) < -0.1:
                num_to_fire = max(1, int(len(performance_scores) * 0.05))
            elif state.get("profit_trend", 0) > 0.1:
                num_to_fire = 0  # don't fire in growth
            else:
                num_to_fire = 1

            # Remove them from the consultants list
            to_fire = set(name for name, _ in worst)
            self.consultants.humans = [h for h in self.consultants.humans if h.name not in to_fire]

            print(f"🔥 Fired 5 lowest-performing consultants: {', '.join(to_fire)}")
            
            # Log to file with more details
            with open("former_consultants.log", "a") as f:
                f.write(f"\n=== Day {self.day_counter} Termination Report ===\n")
                for name, score in worst:
                    history = self.consultants.performance_history.get(name, [])
                    if history:
                        recent = history[-1]
                        f.write(f"Day {self.day_counter}, Fired: {name}, Score: {score:.2f}, "
                                f"Recent Revenue: ${recent['revenue']}, "
                                f"Valid Decision Ratio: {recent['valid_ratio']:.2f}\n")
                    else:
                        f.write(f"Day {self.day_counter}, Fired: {name}, Score: {score:.2f} (No history)\n")

        return len(self.consultants.humans)
    
# --- Security ---
class Security(IRSAgent):
    def __init__(self):
        super().__init__("Security & Personnel")
        self.add_humans(5, "Security", 2_000)

    def act(self, state, context=None):
        # Random security events
        if random.random() < 0.05:  # 5% chance of security event
            event_type = random.choice(["audit", "compliance_check", "training"])
            cost = 0
            
            if event_type == "audit":
                cost = random.randint(1000, 5000)
                event_description = f"Security Audit: ${cost} cost"
            elif event_type == "compliance_check":
                cost = random.randint(500, 2000)
                event_description = f"Compliance Check: ${cost} cost"
            elif event_type == "training":
                cost = random.randint(1000, 3000)
                event_description = f"Security Training: ${cost} cost"
            
            self.memory.append({"event": event_type, "cost": cost})
            print(f"🔒 Security Event: {event_description}")
            return {"event": event_type, "cost": cost}
        
        self.memory.append("On-site")
        return {"event": "routine", "cost": 0}

# --- Executive Office ---
class ExecutiveOffice(IRSAgent):
    def __init__(self):
        super().__init__("Executive Office")
        self.salaries = {"CEO": 30_000, "CTO": 25_000, "CFO": 25_000}
        for title, salary in self.salaries.items():
            self.humans.append(Human(name=names.get_full_name(), role=title, salary=salary))

    def act(self, state, context=None):
        total = sum(v / 30 for v in self.salaries.values())
        self.memory.append(total)
        return total

# --- Accountants ---
class Accountants(IRSAgent):
    def __init__(self):
        super().__init__("Accountants")
        self.add_humans(5, "Accountant", 3_000)
        self.profit_history = []

    def act(self, state, context):
        files = context.get("files", [])
        revenue = sum(f["revenue"] for f in files)
        base_cost = 3833  # Fixed simplified daily cost
        
        # Add any security costs
        security_cost = state.get("security_cost", 0)
        cost = base_cost + security_cost
        
        profit = revenue - cost
        
        # Track profit history
        self.profit_history.append(profit)
        
        # Calculate profit trend (over last 30 days if available)
        profit_trend = 0
        if len(self.profit_history) >= 30:
            recent_profits = self.profit_history[-30:]
            if len(recent_profits) > 1:
                # Simple linear regression slope
                x = list(range(len(recent_profits)))
                y = recent_profits
                slope, _ = np.polyfit(x, y, 1)
                profit_trend = slope / max(1, abs(sum(recent_profits)/len(recent_profits)))
        
        self.memory.append({
            "revenue": revenue, 
            "cost": cost, 
            "profit": profit,
            "profit_trend": profit_trend
        })
        
        return revenue, cost, profit, profit_trend

# --- Investors ---
class Investor:
    def __init__(self, name, equity_share):
        self.name = name
        self.equity_share = equity_share
        self.earnings = []

    def receive_dividend(self, profit):
        payout = self.equity_share * profit if profit > 0 else 0
        self.earnings.append(payout)
        return payout

# --- Simulation ---
class ConsultingFirmIRS:
    def __init__(self):
        consultants = Consultants()
        self.s_history = []
        self.agents = {
            'f': consultants,
            'g': HumanResources(consultants),
            'h': Experts(),
            'i': MiddleOffice(),
            'j': Accountants(),
            'k': Security(),
            'exec': ExecutiveOffice()
        }
        self.investors = [Investor(f"Investor {i+1}", 0.2) for i in range(5)]
        self.day = 0
        self.cash_balance = 0
        self.total_hired = 0
        self.total_fired = 0
        self.random_events = []
        self._init_irs_logs()
        
        # Create output directories if they don't exist
        os.makedirs("reports", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)

    def simulate_days(self, total_days=100):
        for _ in range(total_days):
            self.day += 1
            print(f"\n📅 Day {self.day}")
             
            self._process_random_events()

            if self.day % 10 == 0:
                self.agents['f'].update_performance_history(self.day)

            # --- IRS Core Components ---
            files = self.agents['f'].act({"day": self.day})  # f(S): Consultants propose
            f_output = len(files)

            validated = self.agents['h'].act(None, {"files": files})
            g_output = len(validated)

            # Process middle office
            self.agents['i'].act(None, {"files": validated})

            # Process security
            security_result = self.agents['k'].act(None)
            security_cost = security_result.get("cost", 0)

            # Executive costs
            exec_cost = self.agents['exec'].act(None)

            # Accountant computes revenue, cost, profit, trend
            revenue, cost, profit, profit_trend = self.agents['j'].act(
                {"security_cost": security_cost},
                {"files": validated}
            )

            profit -= exec_cost
            self.cash_balance += profit

            # Calculate IRS values after knowing revenue
            f_output = len(files)
            valid_revenue_decisions = sum(1 for file in validated if file["revenue"] > 0)
            C_fg, K = IRSCalculus.compute_coupling(f_output, g_output, valid_revenue_decisions)
            S_next = IRSCalculus.compute_next_state(self.cash_balance, C_fg, K)
            self.s_history.append(S_next)
            IRSCalculus.log_step(self.irs_log_file, self.day, f_output, g_output, valid_revenue_decisions, C_fg, K, S_next)

            # h(S): HR hiring/firing
            before = len(self.agents['f'].humans)
            self.agents['g'].act({"profit": profit, "profit_trend": profit_trend}, None)
            after = len(self.agents['f'].humans)

            if after > before:
                self.total_hired += after - before
            elif before > after:
                self.total_fired += before - after

            # Check for random events
            self._process_random_events()

            # Update consultant performance history every 10 days
            if self.day % 10 == 0:
                self.agents['f'].update_performance_history(self.day)

            # Consultants propose decisions
            files = self.agents['f'].act({"day": self.day})
            print(f"🧠 {len(files)} decisions proposed.")

            # Experts validate decisions
            validated = self.agents['h'].act(None, {"files": files})
            print(f"✅ {len(validated)} decisions validated by Experts.")
            
            # Track cash balance
            self.s_history.append(self.cash_balance)
            
            # Update CSV tracking
            self._update_trajectory_csv()
            
            # Generate plot every 50 days
            if self.day % 50 == 0:
                self._update_trajectory_plot()
                self._generate_consultant_performance_charts()

            # Log consultant decisions
            self._log_consultant_decisions(validated)

            # Process middle office
            self.agents['i'].act(None, {"files": validated})
            
            # Process security - might have costs
            security_result = self.agents['k'].act(None)
            security_cost = security_result.get("cost", 0)

            # Calculate executive costs
            exec_cost = self.agents['exec'].act(None)
            
            # Calculate financial results
            revenue, cost, profit, profit_trend = self.agents['j'].act(
                {"security_cost": security_cost}, 
                {"files": validated}
            )
            
            profit -= exec_cost
            self.cash_balance += profit
            
            # Add today's cash balance
            S_today = self.cash_balance / 30  # Normalized to monthly units
            self.s_history.append(S_today)

            # Process HR actions (hiring/firing)
            before = len(self.agents['f'].humans)
            self.agents['g'].act({
                "profit": profit,
                "profit_trend": profit_trend
            }, None)
            after = len(self.agents['f'].humans)

            if after > before:
                self.total_hired += after - before
            elif before > after:
                self.total_fired += before - after
                
            # Display daily financial summary
            print(f"💰 Revenue: ${revenue} | Costs: ${cost + exec_cost:.0f} | Profit: ${profit}")
            print(f"🏦 Cash Balance: ${self.cash_balance:,.0f}")

            # Process investor dividends
            for investor in self.investors:
                div = investor.receive_dividend(profit)
                print(f"💸 {investor.name} received: ${div:.0f}")
                
            # Create 100-day bilan if needed
            if self.day % 100 == 0:
                self._generate_100_day_bilan()

        # Inside simulate_days method where you generate other plots
            if self.day % 50 == 0:
                self._generate_irs_components_plot()  # Add this line

        print("\n🏁 Simulation terminée.")
        print("📊 Résumé global :")
        print(f"🔁 Total jours simulés : {self.day}")
        print(f"👥 Consultants en fin de simulation : {len(self.agents['f'].humans)}")
        print(f"🧑‍🏫 Total recrutés : {self.total_hired}")
        print(f"🔥 Total licenciés : {self.total_fired}")
        print(f"🏦 Cash final : ${self.cash_balance:,.2f}")
        print("📝 Tous les bilans 100-cycle ont été enregistrés dans bilan_100_global.csv")

    def _process_random_events(self):
        """Process random events that can affect the simulation"""
        if random.random() < 0.02:  # % chance of random event
            event_types = [
                {"name": "Market Downturn", "cash_impact": -random.randint(5000, 15000), "emoji": "📉"},
                {"name": "New Regulation", "cash_impact": -random.randint(2000, 10000), "emoji": "📜"},
                {"name": "Client Lawsuit", "cash_impact": -random.randint(15000, 40000), "emoji": "⚖️"}
            ]
            
            event = random.choice(event_types)
            self.cash_balance += event["cash_impact"]
            
            print(f"{event['emoji']} RANDOM EVENT: {event['name']} (Impact: ${event['cash_impact']:+,})")
            
            # Record the event
            self.random_events.append({
                "day": self.day,
                "name": event["name"],
                "impact": event["cash_impact"]
            })
            
            # Log to event file
            with open("random_events.log", "a") as f:
                f.write(f"Day {self.day}: {event['name']} - Impact: ${event['cash_impact']:+,}\n")

    def _update_trajectory_csv(self):
        """Update the CSV file tracking the cash balance trajectory"""
        with open("irs_trajectory.csv", "w") as f:
            f.write("day,cash_balance\n")
            for day, cash in enumerate(self.s_history, 1):
                f.write(f"{day},{cash}\n")

    def _update_trajectory_plot(self):
        """Update the plot showing the cash balance trajectory"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.s_history) + 1), self.s_history, marker='o', linestyle='-', color='dodgerblue')
        
        # Add event markers if they exist
        for event in self.random_events:
            day = event["day"]
            if day <= len(self.s_history):
                plt.plot(day, self.s_history[day-1], 'ro')
                plt.annotate(event["name"], (day, self.s_history[day-1]), 
                             textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title("Trajectory of S(t): Cash Balance Over Time")
        plt.xlabel("Day")
        plt.ylabel("Cash Balance ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"visualizations/irs_trajectory_plot_day_{self.day}.png")
        plt.close()

    def _log_consultant_decisions(self, validated_files):
        """Log consultant decisions to CSV file"""
        log_path = "consultant_log.csv"
        log_exists = os.path.exists(log_path)
        
        with open(log_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["day", "consultant", "valid", "reviewed", "revenue"])
            
            if not log_exists:
                writer.writeheader()

            # Add visual daily header separator
            file.write(f"# === DAY {self.day} ===\n")
            
            for f in validated_files:
                writer.writerow({
                    "day": self.day,
                    "consultant": f["consultant"],
                    "valid": f["valid"],
                    "reviewed": f["reviewed"],
                    "revenue": f["revenue"]
                })

    def _generate_consultant_performance_charts(self):
        """Generate performance charts for consultants"""
        consultants = self.agents['f']
        performance_data = consultants.performance_history
        
        if not performance_data:
            return
            
        # Create plot for top 5 consultants by revenue
        plt.figure(figsize=(12, 8))
        
        # Get the 5 consultants with most data points
        consultants_with_data = sorted(
            performance_data.keys(),
            key=lambda c: len(performance_data[c]),
            reverse=True
        )[:5]
        
        for consultant in consultants_with_data:
            hist = performance_data[consultant]
            days = [entry["day"] for entry in hist]
            revenues = [entry["revenue"] for entry in hist]
            
            if days and revenues:
                plt.plot(days, revenues, marker='o', linestyle='-', label=consultant)
        
        plt.title("Top Consultant Revenue Over Time")
        plt.xlabel("Day")
        plt.ylabel("Revenue ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"visualizations/consultant_revenue_day_{self.day}.png")
        plt.close()
        
        # Create efficiency plot
        plt.figure(figsize=(12, 8))
        
        for consultant in consultants_with_data:
            hist = performance_data[consultant]
            days = [entry["day"] for entry in hist]
            efficiency = [entry["valid_ratio"] for entry in hist]
            
            if days and efficiency:
                plt.plot(days, efficiency, marker='o', linestyle='-', label=consultant)
        
        plt.title("Consultant Decision Validity Ratio Over Time")
        plt.xlabel("Day")
        plt.ylabel("Valid Decision Ratio")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"visualizations/consultant_efficiency_day_{self.day}.png")
        plt.close()

    def _init_irs_logs(self):
        """Initialize IRS equation logging file"""
        self.irs_log_file = "irs_calculations.log"
        with open(self.irs_log_file, "w") as f:
            f.write("day,f_output,g_output,valid_revenue_decisions,C_fg,K,S_next\n")

    def _log_irs_calculations(self, day, f_output, g_output, valid_revenue_decisions, C_fg, K, S_next):
        """Log the IRS equation components and result"""
        with open(self.irs_log_file, "a") as f:
            f.write(f"{day},{f_output},{g_output},{valid_revenue_decisions},{C_fg:.4f},{K:.4f},{S_next:.4f}\n")

    def _generate_irs_components_plot(self):
        """Generate a plot showing IRS equation components over time"""
        if not os.path.exists(self.irs_log_file):
            return
            
        # Read data
        data = {"day": [], "f_output": [], "g_output": [], "C_fg": [], "K": [], "S_next": []}
        with open(self.irs_log_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 7:
                    continue
                data["day"].append(int(parts[0]))
                data["f_output"].append(int(parts[1]))
                data["g_output"].append(int(parts[2]))
                data["C_fg"].append(float(parts[4]))
                data["K"].append(float(parts[5]))
                data["S_next"].append(float(parts[6]))
        
        # Create and save the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # First subplot: f_output and g_output
        ax1.plot(data["day"], data["f_output"], 'b-', label='f(S): Consultant Decisions')
        ax1.plot(data["day"], data["g_output"], 'g-', label='g(S): Expert Validations')
        ax1.set_ylabel('Count')
        ax1.set_title('IRS Decision Outputs')
        ax1.legend()
        ax1.grid(True)
        
        # Second subplot: C_fg, K, and S_next
        ax2.plot(data["day"], data["C_fg"], 'r-', label='C(f,g): Coupling Function')
        ax2.plot(data["day"], data["K"], 'm-', label='K: Revenue Efficiency')
        ax2.plot(data["day"], data["S_next"], 'k-', label='S_next: System State')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Value')
        ax2.set_title('IRS System Components')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"visualizations/irs_components_day_{self.day}.png")
        plt.close()

    def _generate_100_day_bilan(self):
        """Generate comprehensive 100-day bilan"""
        consultants = self.agents['f'].humans
        files = self.agents['f'].files
        
        # Calculate revenue per consultant
        revenue_map = defaultdict(int)
        decisions_map = defaultdict(int)
        valid_decisions_map = defaultdict(int)
        
        for f in files:
            revenue_map[f["consultant"]] += f["revenue"]
            decisions_map[f["consultant"]] += 1
            if f["valid"]:
                valid_decisions_map[f["consultant"]] += 1
        
        # Calculate earnings and metrics for all consultants
        all_metrics = []
        for c in consultants:
            revenue = revenue_map.get(c.name, 0)
            decisions = decisions_map.get(c.name, 0)
            valid_decisions = valid_decisions_map.get(c.name, 0)
            
            # Calculate metrics
            efficiency = valid_decisions / max(1, decisions)
            
            all_metrics.append({
                "name": c.name,
                "revenue": revenue,
                "decisions": decisions,
                "valid_decisions": valid_decisions,
                "efficiency": efficiency,
                "revenue_per_decision": revenue / max(1, decisions)
            })
        
        # Sort by revenue
        revenue_sorted = sorted(all_metrics, key=lambda x: x["revenue"], reverse=True)
        top_earner = revenue_sorted[0] if revenue_sorted else {"name": "None", "revenue": 0}
        worst_5 = sorted(all_metrics, key=lambda x: x["revenue"])[:5]
        
        # Print to console
        print("\n📊 🔁 100-CYCLE BILAN:")
        print(f"👥 Total consultants: {len(consultants)}")
        print(f"🧑‍🏫 Recruited: {self.total_hired} | 🔥 Fired: {self.total_fired}")
        print(f"💼 Top earner: {top_earner['name']} with ${top_earner['revenue']}")
        print("🔻 Bottom 5 earners:")
        for m in worst_5:
            print(f"- {m['name']}: ${m['revenue']}")
        print(f"🏦 Cash Balance: ${self.cash_balance:,.0f}")
        print("─────────────────────────────")
        
        # Save to CSV report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/100day_bilan_day{self.day}_{timestamp}.csv"
        
        with open(report_filename, 'w', newline='') as f:
            fieldnames = ["name", "revenue", "decisions", "valid_decisions", 
                          "efficiency", "revenue_per_decision"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in revenue_sorted:
                writer.writerow(metric)
                
        # Generate summary report in markdown format
        report_md = f"reports/100day_summary_day{self.day}_{timestamp}.md"
        with open(report_md, 'w') as f:
            f.write(f"# 100-Day Bilan Summary - Day {self.day}\n\n")
            f.write(f"**Date generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Organization Overview\n")
            f.write(f"- **Total consultants:** {len(consultants)}\n")
            f.write(f"- **Total recruited:** {self.total_hired}\n")
            f.write(f"- **Total fired:** {self.total_fired}\n")
            f.write(f"- **Current cash balance:** ${self.cash_balance:,.2f}\n\n")
            
            f.write("## Performance Highlights\n")
            f.write(f"### Top Performer\n")
            f.write(f"- **Name:** {top_earner['name']}\n")
            f.write(f"- **Total revenue:** ${top_earner['revenue']:,.2f}\n")
            f.write(f"- **Decisions made:** {top_earner['decisions']}\n")
            f.write(f"- **Valid decision ratio:** {top_earner['efficiency']:.2%}\n\n")
            
            f.write("### Bottom 5 Performers\n")
            for i, m in enumerate(worst_5, 1):
                f.write(f"#### {i}. {m['name']}\n")
                f.write(f"- **Revenue:** ${m['revenue']:,.2f}\n")
                f.write(f"- **Valid decision ratio:** {m['efficiency']:.2%}\n")
            
            f.write("\n## Random Events Summary\n")
            recent_events = [e for e in self.random_events if e["day"] > self.day - 100]
            if recent_events:
                for event in recent_events:
                    f.write(f"- Day {event['day']}: {event['name']} (Impact: ${event['impact']:+,.2f})\n")
            else:
                f.write("- No significant events in this period\n")
        
        print(f"📋 Detailed reports saved to {report_filename} and {report_md}")

        # Append to global bilan CSV
        global_csv = "bilan_100_global.csv"
        global_exists = os.path.exists(global_csv)

        with open(global_csv, 'a', newline='') as f:
            fieldnames = ["day", "name", "revenue", "decisions", "valid_decisions", 
                          "efficiency", "revenue_per_decision"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not global_exists:
                writer.writeheader()

            for metric in revenue_sorted:
                writer.writerow({
                    "day": self.day,
                    **metric
                })

# --- IRSCalculus Module ---
class IRSCalculus:
    @staticmethod
    def compute_coupling(f_output, g_output, valid_revenue):
        C_fg = valid_revenue / max(1, f_output)
        K = valid_revenue / max(1, g_output)
        return C_fg, K

    @staticmethod
    def compute_next_state(cash_balance, C_fg, K):
        return (cash_balance / 30) + K * C_fg

    @staticmethod
    def log_step(file_path, day, f_output, g_output, valid_revenue, C_fg, K, S_next):
        with open(file_path, "a") as f:
            f.write(f"{day},{f_output},{g_output},{valid_revenue},{C_fg:.4f},{K:.4f},{S_next:.4f}\n")

if __name__ == "__main__":
    print("🚀 Starting Enhanced IRS Institutional Simulation...\n")
    demo_firm = ConsultingFirmIRS()
    demo_firm.simulate_days(30000)

    print("\n📁 Consultant File Logs Snapshot:")
    consultants = demo_firm.agents['f']
    for idx, file in enumerate(consultants.files[-10:], 1):
        print(f"{idx}. Consultant={file['consultant']}, Valid={file['valid']}, Reviewed={file['reviewed']}, Revenue=${file['revenue']}")