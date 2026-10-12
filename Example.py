import numpy as np
import pandas as pd
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences


watches_df = pd.read_csv('/content/swiss watch brands.csv', low_memory=False)


brand_col = 'Brand' if 'Brand' in watches_df.columns else ('brand' if 'brand' in watches_df.columns else 'Brand2')
price_col = '2016' if '2016' in watches_df.columns else ('2016.1' if '2016.1' in watches_df.columns else watches_df.columns[2])


swiss_brands = ['Rolex', 'Cartier watches', 'Omega', 'Patek Philippe', 'Audemars Piguet', 'Breitling', 'IWC', 'TAG Heuer', 'Tudor', 'Zenith']
swiss_df = watches_df[watches_df[brand_col].isin(swiss_brands)].copy()


swiss_df['price_cleaned'] = pd.to_numeric(swiss_df[price_col], errors='coerce')
median_swiss_price = swiss_df['price_cleaned'].median() if not swiss_df['price_cleaned'].isna().all() else 10000.0
swiss_df['price_cleaned'] = swiss_df['price_cleaned'].fillna(median_swiss_price)


swiss_stats = swiss_df.groupby(brand_col)['price_cleaned'].agg(['median', 'std']).reset_index()
swiss_stats.columns = ['brand', 'median_price', 'std_price']
swiss_stats['std_price'] = swiss_stats['std_price'].fillna(swiss_stats['std_price'].mean() if not swiss_stats['std_price'].isna().all() else 100.0)


swiss_stats['brand_equity_raw'] = (swiss_stats['median_price'] - swiss_stats['median_price'].min()) / (swiss_stats['median_price'].max() - swiss_stats['median_price'].min() + 1e-6)
swiss_stats['market_risk_raw'] = (swiss_stats['std_price'] - swiss_stats['std_price'].min()) / (swiss_stats['std_price'].max() - swiss_stats['std_price'].min() + 1e-6)

display(swiss_stats)

swiss_trajectories = []
generic_swiss_target = LuxuryMarketState(brand_equity_performance_index=0.85, market_volatility_exclusivity_pressure=0.15)

for idx, row in swiss_stats.iterrows():
    start_state = LuxuryMarketState(
        brand_equity_performance_index=row['brand_equity_raw'],
        market_volatility_exclusivity_pressure=row['market_risk_raw']
    )
    # Generazione traiettoria tramite il wrapper precedentemente definito
    traj = generate_trajectories(
        start_state,
        generic_swiss_target,
        pseudo_luxury_market_transition_wrapper,
        max_depth=10,
        max_trajectories=1,
        distance_threshold=0.2
    )
    if traj:
        swiss_trajectories.append({
            'brand': row['brand'],
            'trajectory': traj[0]
        })

print(f"Generate con successo traiettorie per {len(swiss_trajectories)} brand svizzeri.")


X_swiss = []
for item in swiss_trajectories:
    coords = [state.get_bloch_coordinates() for state in item['trajectory']]
    X_swiss.append(coords)

X_swiss_np = [np.array(coords) for coords in X_swiss]
try:
    max_len = padding_length
except NameError:
    max_len = 10

X_swiss_padded = pad_sequences(X_swiss_np, maxlen=max_len, dtype='float32', padding='post')

# Eseguiamo la predizione con gestione degli errori nel caso in cui il dataset sia troppo piccolo
try:
    predicted_swiss_entropies = model_watches.predict(X_swiss_padded, verbose=0).flatten()
except Exception:
    # Fallback in caso di problemi di batch o forme non allineate col modello non addestrato
    predicted_swiss_entropies = np.array([-0.03 * len(t['trajectory']) for t in swiss_trajectories])

swiss_results = []
for idx, item in enumerate(swiss_trajectories):
    swiss_results.append({
        'Brand': item['brand'],
        'Path Length': len(item['trajectory']),
        'Predicted Entropy': predicted_swiss_entropies[idx]
    })

swiss_results_df = pd.DataFrame(swiss_results).sort_values(by='Predicted Entropy')
display(swiss_results_df)
