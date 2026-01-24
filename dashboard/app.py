"""
Red Cell Theory SIR Simulator Dashboard
v1.0.0 — Interactive SIR model for epidemiological visualization.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive

# SIR model functions
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def simulate_sir(beta, gamma, I0, N, days=200):
    t = np.linspace(0, days, days + 1)
    y0 = [N - I0, I0, 0]
    sol = odeint(sir_model, y0, t, args=(beta, gamma, N))
    return t, sol

# UI: Correct layout structure
app_ui = ui.page_fluid(
    ui.h1("                                                                                                                                   SIR Epidemic Simulator"),
    ui.p(
        "Inspired by 3Blue1Brown's 'Simulating an epidemic'. "
        "Tweak R₀ = β/γ and watch outbreaks unfold!"
    ),
    ui.layout_sidebar(
        ui.sidebar(  # ← CORRECT: ui.sidebar()
            ui.h3("Parameters"),
            ui.input_slider("beta", "Infection rate (β)", 0.05, 0.5, 0.3, step=0.01),
            ui.input_slider("gamma", "Recovery rate (γ)", 0.05, 0.2, 0.1, step=0.01),
            ui.input_slider("I0", "Initial infected", 1, 50, 10),
            ui.input_slider("N", "Population (N)", 1000, 10000, 5000),
            ui.input_slider("days", "Simulation days", 100, 500, 200),
            ui.output_text_verbatim("r0_display"),
        ),
        ui.output_plot("sir_plot"),
        ui.output_ui("explanation")
    )
)

# Server
def server(input, output, session):
    @reactive.Calc
    def params():
        return {
            "beta": input.beta(),
            "gamma": input.gamma(),
            "I0": input.I0(),
            "N": input.N(),
            "days": input.days()
        }

    @output
    @render.text
    def r0_display():
        p = params()
        r0 = p["beta"] / p["gamma"]
        status = "⚠️ High" if r0 > 1.5 else "✅ Controlled" if r0 < 1.2 else "⚡ Explosive"
        return f"R₀ = {r0:.2f} {status}"

    @output
    @render.plot
    def sir_plot():
        p = params()
        t, sol = simulate_sir(**p)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, sol[:, 0], label="Susceptible", linewidth=3, color="blue")
        ax.plot(t, sol[:, 1], label="Infected", linewidth=3, color="red")
        ax.plot(t, sol[:, 2], label="Recovered", linewidth=3, color="green")
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        r0 = p["beta"] / p["gamma"]
        ax.set_title(f"SIR Model (R₀ = {r0:.2f})")
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def explanation():
        p = params()
        t, sol = simulate_sir(**p)
        peak_day = int(np.argmax(sol[:, 1]))
        peak_infected = int(np.max(sol[:, 1]))
        return ui.div(
            f"🔍 Peak infection: Day {peak_day}, {peak_infected:,} cases. "
            f"Total pop: {p['N']:,}, initial cases: {p['I0']}.",
            class_="mt-3 p-3 bg-light rounded"
        )

app = App(app_ui, server)
