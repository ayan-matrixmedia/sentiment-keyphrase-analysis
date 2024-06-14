import subprocess

def run_terminal_commands():
    try:
        # Run terminal commands
        subprocess.run(["pip","install","-r","requirements.txt"])
        subprocess.run(["python","-m", "spacy","download","en_core_web_sm"])
        subprocess.run(["python","-m","spacy","download","en"])
    except Exception as e:
        print(f"Error occurred: {e}")
run_terminal_commands()



import streamlit as st 
import pandas as pd 
from analysis import SentimentPhrases
st.set_page_config(page_title="Sentiment Key-phrase Analysis",page_icon="asset/logo.png")
def main():
    st.title("Sentiment and Key-Phrase Analysis")
    st.subheader("Upload csv file from dataset folder in project directory")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if st.button("Submit"):
        predictor = SentimentPhrases(uploaded_file)
        df = predictor.getPrediction()
        if df is not None:
            st.success("File uploaded successfully!")
            st.write("Uploaded DataFrame:")
            st.write(df)
            csv_file = df.iloc[:,[0,1,2,4,3]].to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_file, file_name="output.csv", mime="text/csv")

# Run the Streamlit app
if __name__ == "__main__":
    main()

