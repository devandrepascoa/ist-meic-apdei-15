import subprocess
import time
import sys

def time_command(command):
    start_time = time.time()
    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        sys.exit(1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python time_script.py [command]")
        sys.exit(1)

    command = ' '.join(sys.argv[1:])
    time_command(command)
