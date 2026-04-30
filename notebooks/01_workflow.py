"""
Notebook-style workflow. Copy cells into Jupyter if preferred.
"""
from src.data import load_raw_data, clean_adsorption_data
from src.molecules import add_molecular_descriptors
from src.recommendation import add_recommendations
from src.modeling import train_compare_models, save_model
from src.interpret import get_feature_importance

raw = load_raw_data()
df = clean_adsorption_data(raw)
df = add_molecular_descriptors(df)
df = add_recommendations(df)

print(df.shape)
print(df[["Adsorbent", "Pollutant", "Capacity", "Solution pH", "Surface area", "logp", "tpsa", "recommended_adsorbent"]].head())

best_model, results, test_data = train_compare_models(df)
print(results)
print(get_feature_importance(best_model).head(20))
save_model(best_model)
