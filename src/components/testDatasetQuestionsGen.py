import json
import traceback
from src.load_config import LoadConfig
from src.components.postgreSQLWrapper import PostgreSQLWrapper
from src.components.openAIWrapper import OpenAIWrapper


def get_docs(postgresql_wrapper: PostgreSQLWrapper, sql_query: str) -> str:
    """
    Fetches records from the PostgreSQL database, filters them to 'id', 'source', and 'text', and returns them as a JSON string.
    """
    records_to_process = []

    try:
        with postgresql_wrapper.connection.cursor() as cursor:
            cursor.execute(sql_query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows_tuples = cursor.fetchall()
            else:
                columns = []
                rows_tuples = []

        if rows_tuples:
            for row_tuple in rows_tuples:
                record_dict = dict(zip(columns, row_tuple))

                # Filter to only 'id', 'source', and 'text'
                filtered_record = {
                    "id": record_dict.get("id", None),
                    "source": record_dict.get("source", None),
                    "text": record_dict.get("text", None),
                }
                records_to_process.append(filtered_record)

            # Convert the list of filtered dictionaries into a JSON string
            return json.dumps(records_to_process, indent=2, ensure_ascii=False)
        else:
            print(f"No records found in .")
            return "[]"  # Return an empty JSON array if no records

    except Exception as e:
        print(f"Error fetching and preparing records: {e}")
        traceback.print_exc()
        return "[]"  # Return empty JSON array on error


if __name__ == "__main__":
    config_loader = LoadConfig()
    postgresql_wrapper = None

    try:
        openai_wrapper = OpenAIWrapper()
        postgresql_wrapper = PostgreSQLWrapper(pg_table=config_loader.pg_table)

        # Fetch and prepare the documents as a JSON string
        sql_query = f"""SELECT id, source, text FROM "{config_loader.pg_table}" WHERE source = 'https://www.gesetze-im-internet.de/enefg/BJNR1350B0023.html'"""
        docs_json_string = get_docs(postgresql_wrapper, sql_query)

        number_of_questions = 10
        final_prompt = config_loader.dataset_gen_prompt_3.format(
            docs=docs_json_string, number_of_questions=number_of_questions
        )

        # print("\n--- Generated Prompt with Docs ---")
        # print(final_prompt)

        oprnai_wrapper = OpenAIWrapper()
        questions = oprnai_wrapperrag_text_generator(final_prompt)
        print("\n--- Generated Questions ---")
        print(questions)

        is_save = True
        if is_save:
            # Save the generated questions to a file
            with open("generated_questions.json", "w", encoding="utf-8") as file:
                file.write(questions)
            print("Generated questions saved to 'generated_questions.json'.")

    except Exception as e:
        print(f"An unexpected error occurred during prompt generation: {e}")
        traceback.print_exc()
    finally:
        if postgresql_wrapper:
            postgresql_wrapper.close_connection()
