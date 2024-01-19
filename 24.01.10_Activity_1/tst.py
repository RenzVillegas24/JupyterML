import pandas as pd

# Example DataFrame with GPU names
data = {'GPU Model': ['Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD','Intel Integrated UHD', 'Intel Integrated UHD', 'Intel UHD', 'Intel Integrated UHD Graphics',
                      'AMD Radeon Vega', 'AMD Radeon Integrated GPU Vega', 'AMD Vega',
                      'NVIDIA GeForce GTX 1060', 'NVIDIA GTX 1050', 'NVIDIA GTX 1070']}
df = pd.DataFrame(data)

# Function to find the closest matching GPU model
def find_closest_match(model, models_list, similarity_threshold=0.8):
    matches = []
    
    for candidate in models_list:
        similarity = sum(a == b for a, b in zip(model, candidate)) / max(len(model), len(candidate))
        if similarity >= similarity_threshold:
            matches.append(candidate)
    
    return matches

# Find the closest matching GPU models and group them
similarity_threshold = 0.8
grouped_models = {}

for index, row in df.iterrows():
    model = row['GPU Model']
    
    # Check if the model is already grouped
    if model not in grouped_models:
        matches = find_closest_match(model, df['GPU Model'], similarity_threshold)
        grouped_models[model] = matches

# Create a mapping from each model to its group
model_to_group = {model: group[0] for group in grouped_models.values() for model in group}

# Map the DataFrame to the unified model names
df['Unified Model'] = df['GPU Model'].map(model_to_group)

# Display the DataFrame
print(df)
