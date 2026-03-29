import os

from dotenv import load_dotenv

from analytics_core import (
    build_enriched_dataset,
    exportable_columns,
    fetch_current_weather,
    format_visibility,
    persist_snapshot,
)


load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")


def main() -> None:
    current_df = fetch_current_weather(API_KEY)
    enriched_df = build_enriched_dataset(current_df, API_KEY)
    persist_snapshot(enriched_df)

    display_df = enriched_df[list(exportable_columns())].copy()
    display_df["Visibility"] = display_df["Visibility"].apply(format_visibility)
    print("--- Punjab Weather Advisory Snapshot ---")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
