import psycopg2
import torch
from src.logger import logger
from src.load_config import LoadConfig
from pgvector.psycopg2 import register_vector
from langchain.docstore.document import Document


class PostgreSQLWrapper:

    def __init__(self, pg_table: str = None):
        config_loader = LoadConfig()

        # self.host = config_loader.pg_host
        # self.port = config_loader.pg_port
        # self.database = config_loader.pg_database
        # self.table = pg_table or config_loader.pg_table
        # self.user = config_loader.pg_user
        # self.password = config_loader.pg_password

        self.host = "localhost"
        self.port = "5432"
        self.database = "ecopilot"
        self.user = "postgres"
        self.password = "postgres"
        self.table = "corpus"
        self.connection = self._connect_to_postgres()

        # self._create_table_if_not_exists()

    def _connect_to_postgres(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            # register_vector(conn)  # Register pgvector extension
            return conn
        except Exception as e:
            logger.error0(f"Error connecting to PostgreSQL: {e}")
            raise

    def close_connection(self):
        """Close the PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed.")

    def _create_table_if_not_exists(self):
        """Create the table if it doesn't exist."""
        try:
            with self.connection.cursor() as cursor:
                # Check if the table already exists
                cursor.execute(
                    f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                    """,
                    (self.table,),
                )
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    # Create the table if it doesn't exist
                    cursor.execute(
                        f"""
                        CREATE TABLE "{self.table}" (
                            id SERIAL PRIMARY KEY,
                            source TEXT,
                            chunk_num INTEGER,
                            embedding VECTOR(1536),
                            text TEXT
                        );
                        """
                    )
                    self.connection.commit()
                    logger.info(f"Table '{self.table}' created successfully.")
                else:
                    logger.info(
                        f"Table '{self.table}' already exists. Skipping creation."
                    )
        except Exception as e:
            logger.error(f"Error creating or checking table: {e}")
            raise

    def run_query(self, query, params=None):
        """
        Run a custom SQL query on the PostgreSQL database.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())

                # Only SELECT and similar queries have descriptions
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    self.connection.commit()  # Required for INSERT/UPDATE/DELETE
                    return [{"status": "success", "rows_affected": cursor.rowcount}]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def insert_embeddings(self, documents, table="corpus"):
        """Insert documents and their embeddings into PostgreSQL."""
        try:
            with self.connection.cursor() as cursor:
                for doc in documents:
                    embedding = doc.metadata.get("embedding")
                    embedding = (
                        embedding.tolist()
                        if isinstance(embedding, torch.Tensor)
                        else embedding
                    )

                    cursor.execute(
                        f"""
                        INSERT INTO "{table}" (source, chunk_num, embedding, text)
                        VALUES (%s, %s, %s, %s);
                    """,
                        (
                            doc.metadata.get("source", "Unknown Source"),
                            doc.metadata.get("chunk_num", 0),
                            embedding,
                            doc.page_content,
                        ),
                    )
                self.connection.commit()
                logger.info(f"Inserted {len(documents)} embeddings into PostgreSQL.")
        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            raise

    def get_record_count(self, table="corpus"):
        """
        Retrieve the total number of records (rows) in the table.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                total_rows = cursor.fetchone()[0]
                return total_rows
        except Exception as e:
            logger.error(f"Error retrieving record count: {e}")
            raise

    def delete_records(self, record_ids=None, table="corpus"):
        """
        Delete records from the table.
        If `record_ids` is provided, delete only those records.
        If `record_ids` is None, delete all records.
        """
        try:
            with self.connection.cursor() as cursor:
                if record_ids:
                    # Delete specific records by IDs
                    record_ids_str = ",".join(
                        map(str, record_ids)
                    )  # Convert IDs to a comma-separated string
                    cursor.execute(
                        f"""DELETE FROM "{table}" WHERE id IN ({record_ids_str});"""
                    )
                    logger.info(f"Deleted {len(record_ids)} records from PostgreSQL.")
                else:
                    # Delete all records
                    cursor.execute(f"""DELETE FROM "{table}";""")
                    logger.info("Deleted all records from PostgreSQL.")
                self.connection.commit()
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            raise


if __name__ == "__main__":
    # Load the configuration
    config_loader = LoadConfig()
    dummy_docs = [
        Document(
            page_content="This is a test document about AI and machine learning.",
            metadata={"source": "source1", "chunk_num": 1},
        ),
    ]

    # Initialize PostgreSQLWrapper
    postgresql_wrapper = PostgreSQLWrapper()

    # Insert dummy documents into PostgreSQL
    # postgresql_wrapper.insert_embeddings(dummy_docs)

    # Get the total number of records in the table
    total_records = postgresql_wrapper.get_record_count()
    print(f"Total records in the table: {total_records}")

    # Delete all records from the table
    # postgresql_wrapper.delete_records(table=config_loader.pg_table)

    # Close the PostgreSQL connection
    postgresql_wrapper.close_connection()
