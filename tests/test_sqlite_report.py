import sqlite3

def generate_sqlite_report(db_path):
    """Generate a report of the number of tables and entries per table in the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get the list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        report = []
        report.append(f"Number of tables: {len(tables)}\n")

        # Count entries in each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            report.append(f"Table: {table_name}, Entries: {count}\n")

        conn.close()
        return "".join(report)

    except sqlite3.Error as e:
        return f"Error querying the database: {e}"

if __name__ == "__main__":
    # Path to the SQLite database
    db_path = "storage/sqlite/database.sqlite3"  # Update this path if needed

    report = generate_sqlite_report(db_path)
    print(report)