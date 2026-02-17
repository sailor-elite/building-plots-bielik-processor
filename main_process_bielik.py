# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Config

import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import re

import tomllib
import paramiko
from scp import SCPClient
import os
from datetime import datetime
import glob

TABLE_NAME = "fourlomza_raw"
#MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct-FP8-Dynamic"
MODEL_ID = r"bielik_v3_local"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TOKENS = 256
BATCH_SIZE = 16
SYSTEM_PROMPT = (
    "Jesteś precyzyjnym asystentem do parsowania ogłoszeń. Twoim jedynym zadaniem jest **konwersja ogłoszenia na format JSON** wywołania funkcji. "
    "Odpowiedź musi być **ZAWSZE i WYŁĄCZNIE** poprawnym i kompletnym JSON-em funkcji, bez żadnych dodatkowych komentarzy. "
    "**ODPOWIEDŹ MUSI SIĘ ZACZYNAĆ TYLKO OD ZNAKU '{'**."
    "**Wymagany format JSON to ZAWSZE:** "
    "{"
    "'name': 'parse_offer' lub inna nazwa,"
    "'arguments': {"
    "'OFFER_TYPE': '...' (Wartości: 'WYNAJEM', 'SPRZEDAZ', 'KUPNO'),"
    "'ESTATE_TYPE': '...' (Wartości: 'MIESZKANIE','POKÓJ', 'DOM', 'DZIAŁKA BUDOWLANA', 'DZIAŁKA ROLNA', 'GARAŻ', 'LOKAL', 'LAS', 'INNE'),"
    "'PRICE': '...' (Wartość jako **CZYSTA LICZBA CAŁKOWITA**, bez symboli walut, przecinków, kropek, ani operacji matematycznych),"
    "'AREA_M2': '...' (Wartość jako **CZYSTA LICZBA CAŁKOWITA**."
    "'LOCATION': '...' (Nazwa Miejscowości, Adres lub Ulica)"
    "}"
    "}"
    "Jeśli jakakolwiek informacja **nie jest** wyraźnie podana w tekście, wpisz dla niej wartość: **BRAK**. W PRZYPADKU WĄTPLIWOŚCI CO DO LICZBOWEJ WARTOŚCI (PRICE LUB AREA_M2), ZAWSZE UŻYWAJ **BRAK**."
)


TABLE_PROCESSED = "analyzed_offers"

with open("secrets.toml", "rb") as f:
    config = tomllib.load(f)
creds = config["mikrus"]

OUTPUT_FOLDER = "output"

NEW_NAME = "olx.db"


# # Helper functions

def load_data_to_df(db_name: str, table_name: str) -> pd.DataFrame | None:
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        print(f"DB connected: {db_name}")

        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        print(f"DF loaded: {table_name}")
        print(f"DF shape: {df.shape}")
        return df

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if conn:
            conn.close()
            print("DB connection closed.")


def process_texts_with_bielik(text_series: pd.Series, index: pd.Index) -> pd.Series:
    print("\n--- Model Setup ---")
    print(f"Device: {DEVICE}")

    print("Tokenizer loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )#.to(DEVICE)

    print(f"\n--- Inference Loop Start (Batch Size: {BATCH_SIZE}) ---")

    analysis_results = pd.Series(index=index, dtype=str)
    all_indices = index.tolist()
    all_texts = text_series.tolist()

    total_generation_time = 0

    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Processing Batches"):
        batch_indices = all_indices[i:i + BATCH_SIZE]
        batch_texts = all_texts[i:i + BATCH_SIZE]

        batch_prompts = []
        for input_text in batch_texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Przekonwertuj to ogłoszenie na format JSON: {input_text}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)

        inputs = tokenizer(batch_prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=1024).to(model.device)

        start_time = time.time()
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generation_time = time.time() - start_time
        total_generation_time += generation_time

        decoded_outputs = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[-1]:],
                                                 skip_special_tokens=True)

        for idx_in_batch, generated_text in enumerate(decoded_outputs):
            global_index = batch_indices[idx_in_batch]
            analysis_results.loc[global_index] = generated_text.strip()

            #print(f"Row {global_index}: '{batch_texts[idx_in_batch][:40]}...' -> Res: {generated_text.strip()}")

    print(f"\n--- Inference Complete ---")

    return analysis_results


def save_analysis_to_db(df: pd.DataFrame, db_name: str, table_name: str, analysis_col: str):
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        print(f"\nDB connected for saving: {db_name}")


        temp_df = df[[analysis_col]].copy()
        temp_df.to_sql(f"{table_name}_analysis_temp", conn, if_exists='replace', index=True)

        try:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {analysis_col} TEXT")
            print(f"Column {analysis_col} added to {table_name}.")
        except sqlite3.OperationalError:
            pass

        update_query = f"""
        UPDATE {table_name}
        SET {analysis_col} = (
            SELECT {analysis_col}
            FROM {table_name}_analysis_temp
            WHERE {table_name}_analysis_temp.index = {table_name}.index_col 
        )
        WHERE EXISTS (
            SELECT 1
            FROM {table_name}_analysis_temp
            WHERE {table_name}_analysis_temp.index = {table_name}.index_col
        );
        """


        df[[analysis_col]].to_sql(f"{table_name}_ANALYSIS", conn, if_exists='replace', index=True)
        print(f"Analysis results saved successfully to new table: {table_name}_ANALYSIS.")

    except sqlite3.Error as e:
        print(f"SQLite error during save: {e}")
    except Exception as e:
        print(f"Error during save: {e}")
    finally:
        if conn:
            conn.close()
            print("DB connection closed.")


def download_db():
    now = datetime.now().strftime("%Y-%m-%d")
    folder_name = "db"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    local_filename = f"{now}_olx.db"
    local_full_path = os.path.join(folder_name, local_filename)

    print(f"Connecting {creds['host']}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(
            hostname=creds["host"],
            port=creds["port"],
            username=creds["user"],
            password=creds["password"]
        )
        
        with SCPClient(ssh.get_transport()) as scp:
            print(f"Pobieranie {creds['remote_path']} -> {local_full_path}")
            scp.get(creds["remote_path"], local_full_path)
            
        print(f"Database saved as: {local_full_path}")
        return local_full_path  
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        ssh.close()


def get_latest_db_path(folder="db"):
    files = glob.glob(os.path.join(folder, "*.db"))
    
    if not files:
        print("Not found file with extension 'db'!")
        return None
    
    latest_file = max(files)
    print(f"newest file: {latest_file}")
    return latest_file


def upload_db(local_full_path):
    if not local_full_path or not os.path.exists(local_full_path):
        print("Local file doesn't exist.")
        return False

    original_filename = os.path.basename(local_full_path)
    new_filename = NEW_NAME 
    
    remote_upload_path = os.path.join(creds["remote_folder"], new_filename)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=creds["host"],
            port=creds["port"],
            username=creds["user"],
            password=creds["password"]
        )

        with SCPClient(ssh.get_transport()) as scp:
            print(f"Sending {local_full_path} -> {remote_upload_path}")
            scp.put(local_full_path, remote_upload_path)

        print(f"File succcesfully sent: {new_filename}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        ssh.close()


def merge_and_upload(local_full_path):
    master_db = get_latest_db_path()
    
    if not os.path.exists(local_full_path):
        print("Source file does not exist.")
        return

    try:
        conn = sqlite3.connect(master_db)
        cursor = conn.cursor()

        try:
            cursor.execute("DETACH DATABASE source_db")
        except:
            pass
            
        cursor.execute(f"ATTACH DATABASE '{local_full_path}' AS source_db")

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analyzed_offers'")
        table_exists = cursor.fetchone()

        if not table_exists:
            cursor.execute("CREATE TABLE main.analyzed_offers AS SELECT * FROM source_db.analyzed_offers")
        else:
            cursor.execute("INSERT OR IGNORE INTO main.analyzed_offers SELECT * FROM source_db.analyzed_offers")
        
        added_count = conn.total_changes
        conn.commit()
        
        print(f"Success: Processed {added_count} records.")

        cursor.execute("DETACH DATABASE source_db")
        conn.close()

        upload_db(master_db)

    except Exception as e:
        print(f"Error: {e}")


# # Loading data

download_db()

DB_NAME = get_latest_db_path()
main_df = load_data_to_df(DB_NAME, TABLE_NAME)

data_ids = load_data_to_df(DB_NAME, TABLE_PROCESSED)

main_df = main_df[~main_df["ID"].isin(data_ids["ID"].unique())]

len(main_df)

# # Data processing

# +
text_column = main_df['TEXT']
df_index = main_df.index

analysis_series = process_texts_with_bielik(text_column, df_index)
main_df['ANALYSIS_RESULT'] = analysis_series
#save_analysis_to_db(main_df, DB_NAME, TABLE_NAME, 'ANALYSIS_RESULT')
print("\n--- Inference Complete ---")
print(main_df[['TEXT', 'ANALYSIS_RESULT']].head())
print("\nFinal DF ready.")
# -

# ## Clearing output

main_df["RAW_ARGS"] = main_df["ANALYSIS_RESULT"].str.split('"arguments":').str[1].str.split('}').str[0].str.replace('"',"").str.replace('{',"").str.replace('\\',"").str.upper().str.strip()

temp_df = main_df["RAW_ARGS"].str.split(',', expand=True)
for col in temp_df.columns:
    split_data = temp_df[col].str.split(':', expand=True)
    
    col_name = split_data[0].str.strip().iloc[0]
    
    if col_name: 
        main_df[col_name] = split_data[1].str.strip()

# ## Data visualisation

columns = ["OFFER_TYPE","ESTATE_TYPE", "PRICE", "AREA_M2", "LOCATION"]

for col in columns:
    display(main_df[col].value_counts())

# +
now = datetime.now().strftime("%Y-%m-%d")
output_db_path = os.path.join(OUTPUT_FOLDER, f"{now}_olx_output.db")


with sqlite3.connect(output_db_path) as conn:
        main_df.to_sql("analyzed_offers", conn, if_exists="replace", index=False)

# -


# ## Sending data to server

output_db_path = os.path.join(OUTPUT_FOLDER, f"{now}_olx_output.db")

conn = sqlite3.connect(output_db_path)

upload_df = pd.read_sql("SELECT * FROM analyzed_offers", conn)
conn.close()

upload_df = upload_df.drop_duplicates("ID")

conn_clean = sqlite3.connect(output_db_path)
upload_df.to_sql("analyzed_offers", conn_clean, if_exists="replace", index=False)
conn_clean.close()

merge_and_upload(output_db_path)


