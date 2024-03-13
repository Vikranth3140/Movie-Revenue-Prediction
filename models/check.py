import csv

# Open the CSV file
with open('final_dataset.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Iterate over each row and remove $ signs from the 'gross' column
for row in rows:
    if isinstance(row[2], str) and '$' in row[2]:
        row[2] = row[2].replace('$', '')

# Write the modified data back to the CSV file
with open('final_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)