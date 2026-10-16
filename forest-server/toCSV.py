import csv
import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect("forest.db")
cursor = conn.cursor()

# Select the data you want to export
cursor.execute("SELECT * FROM records")

# Open a CSV file for writing
with open("output.csv", "w", newline="", encoding="utf-8") as csv_file:
  writer = csv.writer(csv_file)

  # Write the column headers
  writer.writerow([description[0] for description in cursor.description])

  # Write the data rows
  writer.writerows(cursor.fetchall())

# Close the connection
conn.close()