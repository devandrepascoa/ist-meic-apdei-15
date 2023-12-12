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
    with open("plots/Q2/execution_times.txt", "a") as file:
        file.write(f"{test_name} (Run {i}) - Elapsed time: {elapsed_time:.2f} seconds\n")
    i += 1


def run_mlp_test(test_name):
    command = f"python ./hw1-q1.py mlp"
    run_command(command, test_name)

def run_perceptron_test(test_name):
    command = f"python ./hw1-q1.py perceptron"
    run_command(command, test_name)
 
def run_logistic_regression_test(test_name,learning_rate=0.01):
    command = f"python ./hw1-q1.py logistic_regression -epochs=50 -learning_rate={learning_rate}"
    run_command(command, test_name)


# Perceptron exercise
run_perceptron_test("1.a) Perceptron")

# Logistic Regression exercise
for lr in [0.01, 0.001]:
    run_logistic_regression_test(f"1.b) Logistic Regression - Learning Rate {lr}", learning_rate=lr)

# MLP exercise
run_mlp_test("1.c) MLP")