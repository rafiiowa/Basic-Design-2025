import numpy as np
import matplotlib.pyplot as plt

def create_health_dashboard(df, model, threshold=30):
    engine_units = df['unit_number'].unique()
    sample_engine = np.random.choice(engine_units)
    engine_data = df[df['unit_number'] == sample_engine].copy()
    features = engine_data.drop(['unit_number', 'time_cycles', 'RUL'], axis=1).columns
    X = engine_data[features]
    engine_data['Predicted_RUL'] = model.predict(X)
    plt.figure(figsize=(12, 5))
    plt.plot(engine_data['time_cycles'], engine_data['RUL'], label='Actual RUL')
    plt.plot(engine_data['time_cycles'], engine_data['Predicted_RUL'], label='Predicted RUL', linestyle='--')
    plt.axhline(y=threshold, color='r', linestyle=':', label='Maintenance Threshold')
    plt.legend()
    plt.title(f'Engine #{sample_engine} Health Monitoring')
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.grid(True)
    plt.tight_layout()
    plt.show()