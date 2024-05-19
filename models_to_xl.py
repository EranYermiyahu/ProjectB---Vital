# import openpyxl
import sys
# import re

# def parse_model_data(model_line):
#   """
#   Parses a line from the data file to extract model name, accuracy, and loss.

#   Args:
#       model_line: A line from the data file containing model information.

#   Returns:
#       A dictionary containing model name, accuracy (train, validation, test), and loss,
#       or None if parsing fails.
#   """
#   model_name_match = re.search(r"^([^ ]+)", model_line)  # Extract model name
#   accuracy_match = re.search(r"train_acc: (\d+\.\d+), validation_acc: (\d+\.\d+), test_acc: (\d+\.\d+)", model_line)  # Extract accuracy
#   loss_match = re.search(r"loss: (\d+\.\d+)", model_line)  # Extract loss

#   if not model_name_match:
#     return None

#   model_name = model_name_match.group(1)
#   accuracy = {}
#   if accuracy_match:
#     accuracy["train"] = float(accuracy_match.group(1))
#     accuracy["validation"] = float(accuracy_match.group(2))
#     accuracy["test"] = float(accuracy_match.group(3))
#   loss = None
#   if loss_match:
#     loss = float(loss_match.group(1))

#   return {"model_name": model_name, "accuracy": accuracy, "loss": loss}

# def write_to_excel(filename, data):
#   """
#   Writes the extracted data to an Excel table.

#   Args:
#       filename: The output Excel file name.
#       data: A list of dictionaries containing model name, accuracy, and loss.
#   """
#   wb = openpyxl.Workbook()
#   ws = wb.active
#   ws.cell(row=1, column=1).value = "Model Name"
#   ws.cell(row=1, column=2).value = "Train Accuracy"
#   ws.cell(row=1, column=3).value = "Validation Accuracy"
#   ws.cell(row=1, column=4).value = "Test Accuracy"
#   ws.cell(row=1, column=5).value = "Loss"
#   row = 2
#   for model_data in data:
#     if model_data:
#       ws.cell(row=row, column=1).value = model_data["model_name"]
#       accuracy = model_data.get("accuracy")
#       loss = model_data.get("loss")
#       if accuracy:
#         ws.cell(row=row, column=2).value = accuracy.get("train", "Missing")
#         ws.cell(row=row, column=3).value = accuracy.get("validation", "Missing")
#         ws.cell(row=row, column=4).value = accuracy.get("test", "Missing")
#       else:
#         ws.cell(row=row, column=2).value = "Missing"
#         ws.cell(row=row, column=3).value = "Missing"
#         ws.cell(row=row, column=4).value = "Missing"
#       ws.cell(row=row, column=5).value = loss if loss is not None else "Missing"
#       row += 1
#     else:
#       ws.cell(row=row, column=1).value = "Error parsing line"
#       ws.cell(row=row, column=2).value = "Missing"
#       ws.cell(row=row, column=3).value = "Missing"
#       ws.cell(row=row, column=4).value = "Missing"
#       ws.cell(row=row, column=5).value = "Missing"
#       row += 1
#   wb.save(filename)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <file_path>")
#         sys.exit(1)

#     file_path = sys.argv[1]
#     with open(file_path, 'r') as f:
#         data = [parse_model_data(line.strip()) for line in f]

#     write_to_excel("model_parameters.xlsx", data)

# print("Data written to model_parameters.xlsx")
import csv
import sys

def convert_to_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    data = []
    models = content.split('Model Name is :')[1:]

    for model in models:
        try:
            lines = model.strip().split('\n')
            model_name = lines[0]
            depth = int(next((line.split(':')[1].strip() for line in lines if 'Model Depth is' in line), 0))
            coverage = int(next((line.split(':')[1].strip() for line in lines if 'Coverage is' in line), 0))
            train_acc = float(next((line.split(':')[2].strip() for line in lines if 'Train Results' in line), 0.0))
            val_acc = float(next((line.split(':')[2].strip() for line in lines if 'Validation Results' in line), 0.0))
            test_acc = float(next((line.split(':')[2].strip() for line in lines if 'Test Results' in line), 0.0))

            data.append([model_name, depth, coverage, train_acc, val_acc, test_acc])
        except (ValueError, IndexError) as e:
            print(f"Error processing model '{model_name}': {e}")
            continue

    with open('output.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model Name', 'Model Depth', 'Coverage', 'Train Categorical Accuracy', 'Validation Categorical Accuracy', 'Test Categorical Accuracy']
        writer = csv.writer(csvfile)

        writer.writerow(fieldnames)
        writer.writerows(data)

    print("CSV file 'output.csv' has been created successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        convert_to_csv(file_path)