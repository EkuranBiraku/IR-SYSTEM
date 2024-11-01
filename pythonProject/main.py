import os
import re
import time
import numpy as np
import pandas as pd
from tkinter import ttk
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import RegexTokenizer, LowercaseFilter, IntraWordFilter
from whoosh.query import Wildcard
from window import show_structured_data, show_unstructured_data, show_combined_data
from whoosh.query import Or
data_type = None


# Load the structured employee data (employee records)
employee_data = pd.read_csv('employee_records_1000_fixed.csv')

# Load the unstructured data (resumes and reviews)
unstructured_data = pd.read_csv('relevant_unstructured_resumes_reviews_1000.csv')

# Define a custom analyzer that preserves numbers and allows partial matches
analyzer = RegexTokenizer() | LowercaseFilter() | IntraWordFilter()

# Define the schema for indexing unstructured data with 'stored=True' and the custom analyzer
# Define the schema for indexing unstructured data with 'stored=True' and the custom analyzer
# Define the schema for indexing unstructured data with 'stored=True' and the custom analyzer
schema = Schema(employee_id=ID(stored=True),
                resume_text=TEXT(stored=True, analyzer=analyzer),
                review_text=TEXT(stored=True, analyzer=analyzer))


# Create index directory if it doesn't exist
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# Create index
ix = create_in("indexdir", schema)

# Index the unstructured data (resumes and reviews)
# Create index and index the unstructured data (resumes only)
# Index the unstructured data (both resume_text and review_text)
writer = ix.writer()
for index, row in unstructured_data.iterrows():
    writer.add_document(employee_id=str(row['employee_id']),
                        resume_text=row['resume_text'],
                        review_text=row['review_text'])
writer.commit()



def refresh_table():
    # Clear the Treeview widget
    tree_widget.delete(*tree_widget.get_children())

    # Reload both CSV files for structured and unstructured data
    global employee_data, unstructured_data
    employee_data = pd.read_csv('employee_records_1000_fixed.csv')  # Structured data
    unstructured_data = pd.read_csv('relevant_unstructured_resumes_reviews_1000.csv')  # Unstructured data

    # Rebuild Whoosh index for unstructured data with new content
    global ix
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    for index, row in unstructured_data.iterrows():
        writer.add_document(employee_id=str(row['employee_id']),
                            resume_text=row['resume_text'],
                            review_text=row['review_text'])
    writer.commit()

    # Reset the search box and filters to their default values
    search_entry.delete(0, tk.END)  # Clear the search box
    department_filter.set(department_options[0])  # Reset the department filter
    role_filter.set(role_options[0])  # Reset the role filter
    structured_selected_column.set(structured_columns[0])  # Reset the structured field filter
    unstructured_selected_field.set(unstructured_fields[0])  # Reset the unstructured field filter
    salary_filter.set(salary_options[0])  # Reset the salary filter

    # Repopulate the Treeview based on the current data_type
    if data_type == "structured":
        display_results("structured", employee_data)
    elif data_type == "unstructured":
        unstructured_results = search_unstructured_data("", field="All")
        display_results("unstructured", unstructured_results)
    elif data_type == "combined":
        combined_search()



def search_structured_data(query, column_name, department_filter=None, role_filter=None, salary_filter=None):
    results = employee_data
    if query.strip():
        lemmatized_query = lemmatize_query(query)

        try:
            query_numeric = float(query)
            if column_name == "All":
                mask = (
                    results['name'].str.contains(lemmatized_query, case=False, na=False) |
                    results['department'].str.contains(lemmatized_query, case=False, na=False) |
                    results['role'].str.contains(lemmatized_query, case=False, na=False) |
                    results['employee_id'].astype(str).str.contains(lemmatized_query, case=False, na=False) |
                    results['salary'].astype(str).str.contains(lemmatized_query, case=False, na=False) |
                    results['hire_date'].str.contains(lemmatized_query, case=False, na=False)
                )
            elif column_name in ['employee_id', 'salary']:
                mask = results[column_name].astype(str).str.contains(lemmatized_query, case=False, na=False)
            else:
                mask = results[column_name].str.contains(lemmatized_query, case=False, na=False)
            results = results[mask]
        except ValueError:
            if column_name == "All":
                mask = (
                    results['name'].str.contains(lemmatized_query, case=False, na=False) |
                    results['department'].str.contains(lemmatized_query, case=False, na=False) |
                    results['role'].str.contains(lemmatized_query, case=False, na=False) |
                    results['hire_date'].str.contains(lemmatized_query, case=False, na=False)
                )
            else:
                mask = results[column_name].str.contains(lemmatized_query, case=False, na=False)
            results = results[mask]

    # Apply department filter
    if department_filter and department_filter != "All":
        results = results[results['department'] == department_filter]

    # Apply role filter
    if role_filter and role_filter != "All":
        results = results[results['role'] == role_filter]

    # Apply salary filter
    if salary_filter and salary_filter != "All":
        if salary_filter == "Less than £70,000":
            results = results[results['salary'] < 70000]
        elif salary_filter == "£70,000+":
            results = results[results['salary'] >= 70000]

    return results





lemmatizer = WordNetLemmatizer()

def lemmatize_query(query):
    return lemmatizer.lemmatize(query.lower())

def extract_experience_details_regex(text):
    """
    Extracts the years of experience and the field from the given text using regex.
    """
    # Refined regex pattern to match phrases like "5 years of experience in data science"
    pattern = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years? of experience in ([\w\s]+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        # Normalize words like "two" to "2"
        words_to_numbers = {
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10"
        }
        years = match.group(1).lower()
        years = words_to_numbers.get(years, years)  # Convert words to numbers if applicable
        field = match.group(2).strip()
        return years, field
    else:
        return "N/A", "N/A"
def extract_review_rating(text):
    """
    Extracts the rating from the review text and formats it as "X/10".
    """
    pattern = r"(\d+)/10"
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)}/10"
    return "N/A"





def search_unstructured_data(query, field="review_text"):
    # Map the user-friendly fields to actual index fields
    if field == "Exp. Years":
        field = "resume_text"  # Field to search for experience details
    elif field == "Rating":
        field = "review_text"  # Field to search for ratings

    lemmatized_query = lemmatize_query(query)  # Get lemmatized form of the query
    extended_query = f"*{lemmatized_query}*"

    with ix.searcher() as searcher:
        # Build the query with wildcard for case-insensitive partial matching
        if field == "All":
            # Search in both 'resume_text' and 'review_text' if 'All' is selected
            parsed_query = Or([Wildcard("resume_text", extended_query), Wildcard("review_text", extended_query)])
        else:
            parsed_query = Wildcard(field, extended_query) if query.strip() else None

        unstructured_results = []
        if parsed_query:
            # Specify no limit to the search results
            results = searcher.search(parsed_query, limit=None)  # Set limit=None to get all results
            for result in results:
                years, field_extracted = extract_experience_details_regex(result['resume_text'])
                rating = extract_review_rating(result['review_text'])
                unstructured_results.append((result['employee_id'], years, field_extracted, rating))
        else:
            # If no specific query, retrieve all documents
            for result in searcher.documents():
                years, field_extracted = extract_experience_details_regex(result['resume_text'])
                rating = extract_review_rating(result['review_text'])
                unstructured_results.append((result['employee_id'], years, field_extracted, rating))

    return unstructured_results














def search_only_structured():
    global data_type
    data_type = "structured"
    query = search_entry.get()
    structured_column = structured_selected_column.get()
    dept_filter = department_filter.get()
    role_filter_value = role_filter.get()
    salary_filter_value = salary_filter.get() if salary_filter.get() != "All" else None

    structured_results = search_structured_data(
        query=query,
        column_name=structured_column,
        department_filter=dept_filter if dept_filter != "All" else None,
        role_filter=role_filter_value if role_filter_value != "All" else None,
        salary_filter=salary_filter_value
    )
    display_results(data_type, structured_results)

    # Calculate metrics
    precision, recall, f1_score, total_retrieved, total_relevant, display_time = calculate_precision_recall_f1(
        query, structured_column, dept_filter, role_filter_value, salary_filter_value
    )

    # Get TF-IDF scores for unstructured data corpus (e.g., resume_text)
    corpus = unstructured_data['resume_text'].tolist()
    tfidf_scores = calculate_tfidf(corpus, query)

    # Show results in popup
    show_results_popup(query, precision, recall, f1_score, tfidf_scores, total_retrieved, total_relevant, display_time)


def search_only_unstructured():
    global data_type
    data_type = "unstructured"
    start_time = time.time()  # Start timer

    query = search_entry.get()
    unstructured_field = unstructured_selected_field.get()

    # Perform unstructured search
    unstructured_results = search_unstructured_data(query, field=unstructured_field)
    display_results(data_type, unstructured_results)

    # Calculate metrics
    total_retrieved = len(unstructured_results)
    total_relevant = sum(1 for result in unstructured_results if
                         query.lower() in result[1].lower() or query.lower() in result[2].lower())
    precision = total_relevant / total_retrieved if total_retrieved > 0 else 0.0
    recall = total_relevant / len(unstructured_data) if len(unstructured_data) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    display_time = time.time() - start_time

    # Get TF-IDF scores for the unstructured data corpus
    corpus = unstructured_data['resume_text'].tolist()
    tfidf_scores = calculate_tfidf(corpus, query)

    # Show results in popup
    show_results_popup(query, precision, recall, f1_score, tfidf_scores, total_retrieved, total_relevant, display_time)


def display_results(data_type, results):
    tree_widget.delete(*tree_widget.get_children())
    tree_widget["columns"] = ()

    if data_type == "structured":
        columns = ["employee_id", "name", "department", "role", "salary", "hire_date"]
        tree_widget["columns"] = columns
        for col in columns:
            tree_widget.heading(col, text=col.capitalize())
            tree_widget.column(col, width=100, anchor="center")

        for _, row in results.iterrows():
            tree_widget.insert("", "end", values=list(row))

    elif data_type == "unstructured":
        columns = ["employee_id", "Exp. Years", "Field", "Review Rating"]
        tree_widget["columns"] = columns
        for col in columns:
            tree_widget.heading(col, text=col.capitalize())
            tree_widget.column(col, width=200, anchor="center")

        for result in results:
            tree_widget.insert("", "end", values=(result[0], result[1], result[2], result[3]))

    elif data_type == "combined":
        columns = ["employee_id", "name", "department", "role", "salary", "hire_date", "Exp. Years", "Field", "Review Rating"]
        tree_widget["columns"] = columns
        for col in columns:
            tree_widget.heading(col, text=col.capitalize())
            tree_widget.column(col, width=150, anchor="center")

        for result in results:
            tree_widget.insert("", "end", values=result)


def combined_search():
    global data_type
    data_type = "combined"
    start_time = time.time()  # Start timer

    query = search_entry.get()
    dept_filter = department_filter.get() if department_filter.get() != "All" else None
    role_filter_value = role_filter.get() if role_filter.get() != "All" else None
    salary_filter_value = salary_filter.get() if salary_filter.get() != "All" else None

    # Perform structured search
    structured_results = search_structured_data(
        query=query,
        column_name="All",
        department_filter=dept_filter,
        role_filter=role_filter_value,
        salary_filter=salary_filter_value
    )

    # Debug: Print structured results
    print("Structured Results:")
    print(structured_results)

    # Perform unstructured search based on the query
    unstructured_results = search_unstructured_data(query, "review_text")

    # Debug: Print unstructured results
    print("Unstructured Results:")
    print(unstructured_results)

    # Combine results by matching employee IDs
    combined_results = []

    # Create a dictionary for quick lookup of unstructured results by employee_id
    unstructured_dict = {int(result[0]): result for result in unstructured_results}  # Convert to int

    for index, structured_row in structured_results.iterrows():
        emp_id = structured_row['employee_id']
        if emp_id in unstructured_dict:
            unstructured_row = unstructured_dict[emp_id]
            combined_results.append((
                emp_id,
                structured_row['name'],
                structured_row['department'],
                structured_row['role'],
                structured_row['salary'],
                structured_row['hire_date'],
                unstructured_row[1],  # Exp. Years
                unstructured_row[2],  # Field
                unstructured_row[3]  # Review Rating
            ))

    # Debug: Print combined results
    print("Combined Results:")
    print(combined_results)

    # Display the combined results
    display_results(data_type, combined_results)

    # Calculate metrics for combined results
    total_retrieved = len(combined_results)
    total_relevant = sum(1 for result in combined_results if query.lower() in str(result).lower())
    precision = total_relevant / total_retrieved if total_retrieved > 0 else 0.0
    recall = total_relevant / (len(employee_data) + len(unstructured_data)) if (len(employee_data) + len(
        unstructured_data)) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    display_time = time.time() - start_time

    # Get TF-IDF scores for both structured and unstructured corpus
    combined_corpus = list(employee_data['name'].astype(str)) + list(unstructured_data['resume_text'].astype(str))
    tfidf_scores = calculate_tfidf(combined_corpus, query)

    # Show results in popup
    show_results_popup(query, precision, recall, f1_score, tfidf_scores, total_retrieved, total_relevant, display_time)


def on_row_click(event):
    selected_item = tree_widget.focus()  # Get the selected item
    if selected_item:
        row_data = tree_widget.item(selected_item, "values")  # Fetch row values
        # Ensure correct order: employee_id, resume_text, review_text
        if data_type == "structured":
            show_structured_data(row_data[:6])  # Show first 6 columns for structured data
        elif data_type == "unstructured":
            show_unstructured_data(row_data)  # Show all columns for unstructured data
        elif data_type == "combined":
            show_combined_data(row_data)  # Show all columns for combined data

# GUI Setup with Styling
root = tk.Tk()
root.title("HR Information Retrieval System")
root.geometry("800x600")
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
style.configure("TEntry", font=("Helvetica", 10))
style.configure("TCombobox", padding=5, font=("Helvetica", 10))
style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
style.configure("TButton", font=("Helvetica", 10), padding=5)

# Search entry and search button
search_frame = ttk.Frame(root, padding=10)
search_frame.pack(fill="x")
ttk.Label(search_frame, text="Search Term:").pack(side="left", padx=5)
search_entry = ttk.Entry(search_frame, width=50)
search_entry.pack(side="left", padx=5)



# Department Filter
filter_frame = ttk.Frame(root, padding=10)
filter_frame.pack(fill="x")
ttk.Label(filter_frame, text="Department Filter:").grid(row=0, column=0, padx=5, sticky="w")
department_options = ["All"] + employee_data['department'].unique().tolist()
department_filter = tk.StringVar(value=department_options[0])
ttk.Combobox(filter_frame, textvariable=department_filter, values=department_options).grid(row=0, column=1, padx=5)

# Role Filter
ttk.Label(filter_frame, text="Role Filter:").grid(row=0, column=2, padx=5, sticky="w")
role_options = ["All"] + employee_data['role'].unique().tolist()
role_filter = tk.StringVar(value=role_options[0])
ttk.Combobox(filter_frame, textvariable=role_filter, values=role_options).grid(row=0, column=3, padx=5)

# Structured and Unstructured Field Selection
ttk.Label(filter_frame, text="Structured Field:").grid(row=1, column=0, padx=5, sticky="w")
structured_columns = ["All", "name", "department", "role"]
structured_selected_column = tk.StringVar(value=structured_columns[0])
ttk.Combobox(filter_frame, textvariable=structured_selected_column, values=structured_columns).grid(row=1, column=1, padx=5)

# Update Unstructured Field Selection
ttk.Label(filter_frame, text="Unstructured Field:").grid(row=1, column=2, padx=5, sticky="w")

# Change the displayed options for unstructured fields
unstructured_fields = ["All", "Exp. Years", "Rating"]
unstructured_selected_field = tk.StringVar(value=unstructured_fields[0])
ttk.Combobox(filter_frame, textvariable=unstructured_selected_field, values=unstructured_fields).grid(row=1, column=3, padx=5)

# Salary Filter
ttk.Label(filter_frame, text="Salary Filter:").grid(row=2, column=0, padx=5, sticky="w")
salary_options = ["All", "Less than £70,000", "£70,000+"]
salary_filter = tk.StringVar(value=salary_options[0])
ttk.Combobox(filter_frame, textvariable=salary_filter, values=salary_options).grid(row=2, column=1, padx=5)

# Results Display
columns = ["employee_id", "name", "department", "role", "salary", "hire_date"]
tree_widget = ttk.Treeview(root, columns=columns, show="headings", height=10)
tree_widget.pack(pady=20, fill="both", expand=True)
for col in columns:
    tree_widget.heading(col, text=col.capitalize())
    tree_widget.column(col, anchor="center")
tree_widget.bind("<Double-1>", on_row_click)  # Double-click to open the row in a new window

# Buttons
button_frame = ttk.Frame(root, padding=10)
button_frame.pack(fill="x")
ttk.Button(button_frame, text="Search Structured Data", command=search_only_structured).pack(side="left", padx=10)
ttk.Button(button_frame, text="Search Unstructured Data", command=search_only_unstructured).pack(side="left", padx=10)
ttk.Button(button_frame, text="Combined Search", command=combined_search).pack(side="left", padx=10)
ttk.Button(button_frame, text="Refresh", command=refresh_table).pack(side="left", padx=10)  # Refresh button
def calculate_tfidf(corpus, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    query_tfidf_scores = {}
    for term in query.split():
        if term in feature_names:
            index = feature_names.tolist().index(term)
            query_tfidf_scores[term] = np.mean(tfidf_matrix[:, index].toarray())

    return query_tfidf_scores

def calculate_precision_recall_f1(query, structured_field, department_filter=None, role_filter=None, salary_filter=None):
    # Start timer
    start_time = time.time()

    # (Existing code to filter data based on department, role, and salary filters)
    query_lower = query.lower()
    search_columns = ["name", "department", "role", "salary", "hire_date"]

    # Apply filters
    filtered_data = employee_data
    if department_filter and department_filter != "All":
        filtered_data = filtered_data[filtered_data['department'] == department_filter]
    if role_filter and role_filter != "All":
        filtered_data = filtered_data[filtered_data['role'] == role_filter]
    if salary_filter and salary_filter != "All":
        if salary_filter == "Less than £70,000":
            filtered_data = filtered_data[filtered_data['salary'] < 70000]
        elif salary_filter == "£70,000+":
            filtered_data = filtered_data[filtered_data['salary'] >= 70000]

    # Perform search
    if structured_field != "All":
        retrieved_results = filtered_data[filtered_data[structured_field].astype(str).str.contains(query, case=False, na=False)]
        relevant_results = retrieved_results[retrieved_results[structured_field].astype(str).str.lower() == query_lower]
        total_relevant_in_filtered_data = filtered_data[filtered_data[structured_field].astype(str).str.lower() == query_lower]
    else:
        retrieved_results = filtered_data[filtered_data[search_columns].apply(lambda row: row.astype(str).str.contains(query, case=False, na=False)).any(axis=1)]
        relevant_results = retrieved_results[retrieved_results[search_columns].apply(lambda row: row.astype(str).str.lower().eq(query_lower)).any(axis=1)]
        total_relevant_in_filtered_data = filtered_data[filtered_data[search_columns].apply(lambda row: row.astype(str).str.lower().eq(query_lower)).any(axis=1)]

    # Calculate precision and recall
    relevant_count = len(relevant_results)
    total_retrieved = len(retrieved_results)
    total_relevant = len(total_relevant_in_filtered_data)
    precision = relevant_count / total_retrieved if total_retrieved > 0 else 0.0
    recall = relevant_count / total_relevant if total_relevant > 0 else 0.0

    # Calculate F1 score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate display time
    display_time = time.time() - start_time

    return precision, recall, f1_score, total_retrieved, total_relevant, display_time


def show_results_popup(query, precision, recall, f1_score, tfidf_scores, total_retrieved, total_relevant, display_time):
    # Create popup window
    popup = tk.Toplevel()
    popup.title("Query Results Analysis")
    popup.geometry("600x500")

    # Title for the popup
    tk.Label(popup, text="Query Results Analysis", font=("Helvetica", 14, "bold")).pack(pady=(10, 10))

    # Create a frame for the table
    table_frame = ttk.Frame(popup)
    table_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Define the columns
    columns = ("Metric", "Value")
    table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
    table.heading("Metric", text="Metric")
    table.heading("Value", text="Value")
    table.column("Metric", anchor="w", width=200)
    table.column("Value", anchor="center", width=300)

    # Insert main metrics as rows in the table
    metrics = [
        ("Query", query),
        ("Total Results Retrieved", total_retrieved),
        ("Total Relevant Results", total_relevant),
        ("Precision", f"{precision:.2f}"),
        ("Recall", f"{recall:.2f}"),
        ("F1 Score", f"{f1_score:.2f}"),
        ("Display Time", f"{display_time:.2f} seconds")
    ]
    for metric, value in metrics:
        table.insert("", "end", values=(metric, value))

    # Insert TF-IDF scores as additional rows
    if tfidf_scores:
        table.insert("", "end", values=("TF-IDF Scores", ""), tags=("header",))
        for term, score in tfidf_scores.items():
            table.insert("", "end", values=(f"  {term}", f"{score:.4f}"))
    else:
        table.insert("", "end", values=("TF-IDF Scores", "No relevant terms found"))

    # Add the table to the window
    table.pack(fill="both", expand=True)

    # Style tags for headers
    table.tag_configure("header", font=("Helvetica", 10, "bold"))

    # Close button at the bottom
    tk.Button(popup, text="Close", command=popup.destroy, font=("Helvetica", 10), width=15).pack(pady=(10, 20))
root.mainloop()

