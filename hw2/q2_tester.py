import subprocess
import time
import sys
i = 0


def run_command(command, test_name):
    global i
    start_time = time.time()
    with open(f"stdout {test_name}.txt", "w") as outfile:
        try:
            # Execute the command and redirect output to a file
            subprocess.run(command, shell=True, check=True, stdout=outfile, stderr=outfile)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            sys.exit(1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    with open("outputs/Q2/execution_times.txt", "a") as file:
        file.write(f"{test_name} (Run {i}) - Elapsed time: {elapsed_time:.2f} seconds\n")
    i += 1


def run_q1_test(test_name, learning_rate=0.1):
    command = f"python3 ./hw2-q2.py -epochs=15 -learning_rate={learning_rate} -dropout=0.7 -optimizer=sgd"
    print(command)
    run_command(command, test_name)

def run_q2_test(test_name, learning_rate=0.1):
    command = f"python3 ./hw2-q2.py -epochs=15 -learning_rate={learning_rate} -dropout=0.7 -optimizer=sgd -no_maxpool"
    print(command)
    run_command(command, test_name)

# Exercise 1
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    run_q1_test(f"1. Question 1 - LR {lr}", learning_rate=lr)

# Exercise 2
run_q2_test(f"2. Question 2 - LR {0.01}", learning_rate=0.01)
