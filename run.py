import subprocess

def run_terminal_commands():
    try:
        # Run terminal commands
        subprocess.run(["pip","install","-r","requirements.txt"])
        subprocess.run(["python","-m", "spacy","download","en_core_web_sm"])
        print("Now running the main file...")
        subprocess.run(["streamlit", "run", "app.py"]) 
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_terminal_commands()
