from <your_root>.core.csv_metadata_ingestor import CSVMetadataIngestor
from <your_root>.ingestion.neaps_oracle_client import NeapsOracleClient


# 🔧 CHANGE THESE PATHS
CSV_DIRECTORY_PATH = r"/path/to/csv/files"
PROCESSED_DIRECTORY_PATH = r"/path/to/processed/files"


def main():

    print("\n🚀 Starting metadata ingestion")
    print(f"Source: {CSV_DIRECTORY_PATH}")
    print(f"Processed: {PROCESSED_DIRECTORY_PATH}")

    client = NeapsOracleClient()
    conn = client._get_conn()

    if conn is None:
        print("❌ Oracle DB connection not available")
        return

    try:
        ingestor = CSVMetadataIngestor(conn)

        ingestor.process_directory(
            CSV_DIRECTORY_PATH,
            PROCESSED_DIRECTORY_PATH
        )

        print("\n✅ Metadata ingestion completed successfully")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()