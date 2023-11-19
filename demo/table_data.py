import pandas as pd

data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [10, 25, 15, 30]
}

df = pd.DataFrame(data)
file_path = 'data.json'
df.to_json(file_path, orient='records')
