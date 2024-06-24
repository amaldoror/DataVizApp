import os
import subprocess
import sys


# Function to create a virtual environment
def create_virtualenv(env_name):
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', env_name])
    except subprocess.CalledProcessError:
        print(f"Failed to create virtual environment {env_name}.")
        sys.exit(1)


# Function to install packages from requirements.txt
def install_packages(env_name):
    activate_script = os.path.join(env_name, "Scripts" if os.name == 'nt' else "bin", "activate")
    subprocess.check_call([activate_script, "&&", "pip", "install", "-r", "requirements.txt"])


# Main function
def main():
    env_name = "venv"  # Name of the virtual environment directory
    create_virtualenv(env_name)
    install_packages(env_name)
    print("Setup complete. Virtual environment created and packages installed.")


if __name__ == "__main__":
    main()
