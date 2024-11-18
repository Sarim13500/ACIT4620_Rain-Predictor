import pandas as pd

# Sample data for Oslo weather
data = {
    "Date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"],
    "Temperature": [-3.5, -4.0, -2.1, -1.0, 0.5],
    "Rainfall": [0.2, 0.0, 0.5, 1.2, 0.3],
    "Humidity": [85, 90, 78, 70, 65],
    "Wind Speed": [12.0, 10.5, 8.9, 15.3, 13.2]
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv("rainfall_data.csv", index=False)

print("CSV file 'oslo_weather_data.csv' created successfully.")
