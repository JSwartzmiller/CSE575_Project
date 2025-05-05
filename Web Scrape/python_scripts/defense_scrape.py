import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://hashtagbasketball.com/nba-defense-vs-position"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

#find specific table with all defensive stats
tables = soup.find_all("table", class_="table--statistics")


#Grab the third table on the page 
table = tables[2]

#gather headers for feautures in table
header_cells = table.find_all("th")
headers = [cell.get_text(strip=True) for cell in header_cells]

#gather all the rows of the table that include data
rows = []
for row in table.find_all("tr")[1:]:
    cells = row.find_all(["td", "th"])
    row_data = [cell.get_text(strip=True) for cell in cells]
    if row_data:
        rows.append(row_data)

#convert new gathered data to dataframe
df = pd.DataFrame(rows, columns=headers)
df.to_csv("defensive_matchup_table.csv", index=False)