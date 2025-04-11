from django import forms

class SimulationSettingsForm(forms.Form):
    pop_size = forms.IntegerField(
        label="Population Size",
        min_value=1,
        initial=10,
        help_text="Number of individuals in the population."
    )
    max_iter = forms.IntegerField(
        label="Max Iterations",
        min_value=1,
        initial=100,
        help_text="How many iterations the simulation should run."
    )
    initial_irradiance = forms.FloatField(
        label="Initial Irradiance (W/m²)",
        initial=800,
        help_text="Irradiance in watts per square meter."
    )
    T_value = forms.FloatField(
        label="Temperature (°C)",
        initial=25,
        help_text="Ambient temperature in degrees Celsius."
    )
    # You can add more fields for additional parameters (e.g., for SA, PSO, etc.)
