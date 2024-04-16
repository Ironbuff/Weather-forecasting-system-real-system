import csv

def check_csv_format(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        num_columns = len(header)
        data_types = []

        # Analyze the data types of each column
        for i in range(num_columns):
            column_values = set()
            for row in reader:
                column_values.add(row[i])
            if all(value.isdigit() for value in column_values):
                data_types.append('numeric')
            elif all(value.replace('.', '', 1).isdigit() for value in column_values):
                data_types.append('numeric (with decimal)')
            elif all(value.lower() in {'true', 'false'} for value in column_values):
                data_types.append('boolean')
            else:
                data_types.append('string')

        # Print the analysis results
        print("CSV file structure analysis:")
        print(f"Number of columns: {num_columns}")
        print("Data types of each column:")
        for i, dtype in enumerate(data_types):
            print(f"Column {i+1}: {dtype}")

# Provide the path to your CSV file
csv_file_path = 'evaluation_metrics.csv'
check_csv_format(csv_file_path)
