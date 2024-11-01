import tkinter as tk
from tkinter import ttk
import pandas as pd
import csv
import os


# Load the structured employee data (employee records)
employee_data = pd.read_csv('employee_records_1000_fixed.csv')

# Load the unstructured data (resumes and reviews)
unstructured_data = pd.read_csv('relevant_unstructured_resumes_reviews_1000.csv')  # Ensure this exists and is loaded


def delete_record_from_both_files(employee_id):
    # Delete from the structured data file
    try:
        with open('employee_records_1000_fixed.csv', 'r', newline='') as structured_file:
            structured_rows = list(csv.reader(structured_file))
            structured_header = structured_rows[0]  # Assume first row is the header
            # Filter out rows where employee_id matches, ensuring no blank rows are left
            structured_filtered_rows = [row for row in structured_rows[1:] if row[0] != str(employee_id)]
            print(f"Deleting employee_id {employee_id} from structured data...")

        # Write the filtered rows back to the structured file
        with open('employee_records_1000_fixed.csv', 'w', newline='') as structured_file:
            writer = csv.writer(structured_file)
            writer.writerow(structured_header)
            writer.writerows(structured_filtered_rows)
            print(f"Employee_id {employee_id} successfully deleted from structured data.")

        # Reload the structured data into memory
        global employee_data
        employee_data = pd.read_csv('employee_records_1000_fixed.csv')

    except Exception as e:
        print(f"Error deleting record in structured data: {e}")

    # Delete from the unstructured data file
    try:
        with open('relevant_unstructured_resumes_reviews_1000.csv', 'r', newline='') as unstructured_file:
            unstructured_rows = list(csv.reader(unstructured_file))
            unstructured_header = unstructured_rows[0]  # Assume first row is the header
            # Filter out rows where employee_id matches, ensuring no blank rows are left
            unstructured_filtered_rows = [row for row in unstructured_rows[1:] if row[0] != str(employee_id)]
            print(f"Deleting employee_id {employee_id} from unstructured data...")

        # Write the filtered rows back to the unstructured file
        with open('relevant_unstructured_resumes_reviews_1000.csv', 'w', newline='') as unstructured_file:
            writer = csv.writer(unstructured_file)
            writer.writerow(unstructured_header)
            writer.writerows(unstructured_filtered_rows)
            print(f"Employee_id {employee_id} successfully deleted from unstructured data.")

        # Reload the unstructured data into memory
        global unstructured_data
        unstructured_data = pd.read_csv('relevant_unstructured_resumes_reviews_1000.csv')

    except Exception as e:
        print(f"Error deleting record in unstructured data: {e}")



# Example usage
def delete_record(employee_id, data_type, detail_window):
    confirm = tk.messagebox.askyesno(
        "Confirm Delete",
        "Are you sure you want to delete this record from both structured and unstructured data?"
    )
    if confirm:
        delete_record_from_both_files(employee_id)  # Delete from both files
        tk.messagebox.showinfo("Deleted", f"Record {employee_id} has been deleted from both datasets.")
        detail_window.destroy()

def show_structured_data(row_data):
    detail_window = tk.Toplevel()
    detail_window.title("Structured Data Details")
    detail_window.geometry("500x400")
    detail_window.configure(bg="#e0e0e0")

    column_names = ["Employee ID", "Name", "Department", "Role", "Salary", "Hire Date"]

    employee_id = row_data[0]  # Get employee_id from row_data

    # Form-like layout
    for i, (label_text, value) in enumerate(zip(column_names, row_data)):
        label = ttk.Label(detail_window, text=label_text + ":", font=("Helvetica", 10, "bold"))
        label.grid(row=i, column=0, sticky="e", padx=10, pady=5)
        value_label = ttk.Label(detail_window, text=value, font=("Helvetica", 10))
        value_label.grid(row=i, column=1, sticky="w", padx=10, pady=5)

    # Add Delete Button
    delete_button = ttk.Button(detail_window, text="Delete",
                               command=lambda: delete_record(employee_id, "structured", detail_window))
    delete_button.grid(row=len(row_data), column=0, columnspan=2, pady=10)

    close_button = ttk.Button(detail_window, text="Close", command=detail_window.destroy)
    close_button.grid(row=len(row_data) + 1, column=0, columnspan=2, pady=10)


def show_unstructured_data(row_data):
    detail_window = tk.Toplevel()
    detail_window.title("Unstructured Data Details")
    detail_window.geometry("600x400")
    detail_window.configure(bg="#e0e0e0")

    column_names = ["Employee ID", "Exp. Years", "Field", "Review Rating"]

    employee_id = row_data[0]  # Get employee_id from row_data

    for i, (label_text, value) in enumerate(zip(column_names, row_data)):
        label = ttk.Label(detail_window, text=label_text + ":", font=("Helvetica", 10, "bold"))
        label.grid(row=i, column=0, sticky="ne", padx=10, pady=5)

        value_label = ttk.Label(detail_window, text=value, font=("Helvetica", 10))
        value_label.grid(row=i, column=1, sticky="w", padx=10, pady=5)

    # Add Delete Button
    delete_button = ttk.Button(detail_window, text="Delete",
                               command=lambda: delete_record(employee_id, "unstructured", detail_window))
    delete_button.grid(row=len(row_data), column=0, columnspan=2, pady=10)

    close_button = ttk.Button(detail_window, text="Close", command=detail_window.destroy)
    close_button.grid(row=len(row_data) + 1, column=0, columnspan=2, pady=10)

def show_combined_data(row_data):
    detail_window = tk.Toplevel()
    detail_window.title("Combined Data Details")
    detail_window.geometry("600x600")
    detail_window.configure(bg="#e0e0e0")

    structured_names = ["Employee ID", "Name", "Department", "Role", "Salary", "Hire Date"]
    unstructured_names = ["Exp. Years", "Field", "Review Rating"]

    employee_id = row_data[0]  # Get employee_id from row_data

    structured_frame = ttk.LabelFrame(detail_window, text="Structured Data", padding=10)
    structured_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

    for i, (label_text, value) in enumerate(zip(structured_names, row_data[:6])):
        label = ttk.Label(structured_frame, text=label_text + ":", font=("Helvetica", 10, "bold"))
        label.grid(row=i, column=0, sticky="e", padx=5, pady=5)
        value_label = ttk.Label(structured_frame, text=value, font=("Helvetica", 10))
        value_label.grid(row=i, column=1, sticky="w", padx=5, pady=5)

    unstructured_frame = ttk.LabelFrame(detail_window, text="Unstructured Data", padding=10)
    unstructured_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

    for i, (label_text, value) in enumerate(zip(unstructured_names, row_data[6:])):
        label = ttk.Label(unstructured_frame, text=label_text + ":", font=("Helvetica", 10, "bold"))
        label.grid(row=i, column=0, sticky="ne", padx=5, pady=5)
        value_label = ttk.Label(unstructured_frame, text=value, font=("Helvetica", 10))
        value_label.grid(row=i, column=1, sticky="w", padx=5, pady=5)

    # Add Delete Button
    delete_button = ttk.Button(detail_window, text="Delete",
                               command=lambda: delete_record(employee_id, "combined", detail_window))
    delete_button.grid(row=6, column=0, columnspan=2, pady=10)

    close_button = ttk.Button(detail_window, text="Close", command=detail_window.destroy)
    close_button.grid(row=7, column=0, columnspan=2, pady=10)
