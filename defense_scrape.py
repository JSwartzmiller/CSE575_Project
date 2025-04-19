import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://hashtagbasketball.com/nba-defense-vs-position"  # Replace with the real URL

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Find all tables with the relevant class
tables = soup.find_all("table", class_="table--statistics")

# Make sure there are enough tables
if len(tables) < 3:
    raise Exception(f"Expected at least 3 tables, found {len(tables)}.")

# Select the third table (index 2 since it's 0-based)
table = tables[2]

# Parse headers
header_cells = table.find_all("th")
headers = [cell.get_text(strip=True) for cell in header_cells]

# Parse rows
rows = []
for row in table.find_all("tr")[1:]:  # Skip header row
    cells = row.find_all(["td", "th"])
    row_data = [cell.get_text(strip=True) for cell in cells]
    if row_data:
        rows.append(row_data)

# Convert to DataFrame
df = pd.DataFrame(rows, columns=headers)
df.to_csv("defensive_matchup_table.csv", index=False)
print("✅ Saved: defensive_matchup_table.csv")
