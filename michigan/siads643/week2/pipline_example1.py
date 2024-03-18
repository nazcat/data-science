import argparse
import pandas as pd

def clean_data(file1, file2):
    contact_df = pd.read_csv(file1)
    other_df = pd.read_csv(file2, converters={'birthdate': str})

    # remove line break from address column
    #contact_df['address'] = contact_df['address'].str.replace('\n', ' ')

    # convert birthdate from MMddyyyy to yyyy-MM-dd
    other_df["date"] = other_df["birthdate"].apply(lambda num: f"{num[4:]}-{num[:2]}-{num[2:4]}")

    # delete old birthday column then rename columns for final output
    del other_df['birthdate']
    other_df.columns = ['respondent_id','job','company','birthdate']

    final = pd.merge(contact_df, other_df, on='respondent_id', how='inner')

    return final

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("contact_info_file", help="contact data file (CSV)")
    parser.add_argument("other_info_file", help="other data file (CSV)")
    parser.add_argument("output_file", help="output data file (CSV)")
    args = parser.parse_args()

    contact_info_file = args.contact_info_file
    other_info_file = args.other_info_file
    output_file_path = args.output_file

    # output file into data folder
    df = clean_data(contact_info_file, other_info_file)
    output = df.to_csv(output_file_path, index=False)
