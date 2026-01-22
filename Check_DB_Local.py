import psycopg2
from psycopg2 import Error
from datetime import datetime

# Database config
DB_CONFIG = {
    'host': '54.225.63.242',
    'port': 5432,
    'database': 'visco',
    'user': 'visco_cctv',
    'password': 'Visco@0408'
}


def check_database():
    """Check database connection and print details"""
    connection = None
    cursor = None

    try:
        print("=" * 60)
        print("DATABASE CONNECTION CHECK")
        print("=" * 60)
        print(f"\nConnecting to database...")
        print(f"Host: {DB_CONFIG['host']}")
        print(f"Database: {DB_CONFIG['database']}")
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Establish connection
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        print("\n✓ Connection successful!")
        print("=" * 60)

        # List all tables in the database
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        print(f"\nTABLES IN DATABASE ({len(tables)} found)")
        print("=" * 60)

        if tables:
            for idx, table in enumerate(tables, 1):
                print(f"{idx}. {table[0]}")
        else:
            print("\nNo tables found in the database.")
            return

        # Ask user for table name
        print("\n" + "=" * 60)
        table_name = input("\nEnter the table name to view details: ").strip()

        # Check if table exists
        table_exists = any(table[0].lower() == table_name.lower() for table in tables)

        if not table_exists:
            print(f"\n❌ Table '{table_name}' not found in database.")
            return

        print("\n" + "=" * 60)
        print(f"TABLE: {table_name}")
        print("=" * 60)

        # Get column details
        cursor.execute(f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()

        print(f"\nCOLUMN DETAILS:")
        print("-" * 60)
        for col in columns:
            col_name, data_type, max_length = col
            length_info = f"({max_length})" if max_length else ""
            print(f"  - {col_name}: {data_type}{length_info}")

        # Get all rows from the table
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()

        print("\n" + "=" * 60)
        print(f"ALL ROWS DATA (Total: {len(rows)} rows)")
        print("=" * 60)

        if rows:
            # Get column names
            col_names = [desc[0] for desc in cursor.description]

            # Print column headers
            print("\n" + " | ".join(col_names))
            print("-" * 60)

            # Print all rows
            for row_idx, row in enumerate(rows, 1):
                print(f"\nRow {row_idx}:")
                for col_name, value in zip(col_names, row):
                    print(f"  {col_name}: {value}")
        else:
            print("\nNo data found in this table.")

        print("\n" + "=" * 60)
        print("QUERY COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Error as e:
        print("\n" + "=" * 60)
        print("❌ DATABASE ERROR")
        print("=" * 60)
        print(f"\nError: {e}")
        print(f"Error Type: {type(e).__name__}")

    finally:
        # Close database connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    check_database()