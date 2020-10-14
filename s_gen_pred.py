import os
import sys
import shutil
import subprocess
from sklearn_porter import Porter


def _clean_cp_xnn():
    print("clean cp_xnn folder")
    if os.path.isdir("cp_xnn"):
        shutil.rmtree("cp_xnn")
    if not os.path.isdir("cp_xnn"):
        os.mkdir("cp_xnn")
    if not os.path.isdir("cp_xnn/dat"):
        os.mkdir("cp_xnn/dat")

def _save_model_to_java(model):
    cpm_filename = "cp_xnn/MLPClassifier.java"
    print("preper new trained model  {}".format(cpm_filename))
    if os.path.isfile(cpm_filename):
        os.unlink(cpm_filename)
    porter = Porter(model, language='java')
    output = porter.export()
    with open(cpm_filename, 'w+') as file:
        n = file.write(output)
        print("traned model saved in c: {} len: {}, wlen: {} ".format(cpm_filename, len(output), n))
    if not os.path.isfile(cpm_filename):
        print("Error: no training model saved")
        sys.exit(0)

def _compile_java_to_executable():
    print("compile java to executable class")
    root_dir = os.getcwd()
    working_dir = root_dir + "/cp_xnn"
    os.chdir(working_dir)
    cmds = ["javac", "MLPClassifier.java"]
    stdout = subprocess.check_output(cmds)
    print(stdout)
    os.chdir(root_dir)

def _prepare_test_data(rx_test, skip_ratio):
    print("prepare test dat (for predict)...")
    for i in range(0, len(rx_test)):
        test_case = rx_test[i]
        cpd_filename = "cp_xnn/dat/{}_{:04d}.tdat".format(len(test_case), i)
        if os.path.isfile(cpd_filename):
            os.unlink(cpd_filename)

        with open(cpd_filename, 'w+') as file:
            for j in range(0, len(test_case)):
                if j == len(test_case) - 1:
                    file.write("{:.3f}".format(test_case[j]))
                else:
                    file.write("{:.3f} ".format(test_case[j]))
    print("total {} test prepared in cp_xnn/dat, each has {} features".format(len(rx_test), len(rx_test[0])))

    cpi_filename = "cp_xnn/cp_info.text"
    with open(cpi_filename, 'w+') as file:
        file.write("{}\n".format(len(rx_test)))
        file.write("{}\n".format(len(rx_test[0])))
        file.write("{}\n".format(skip_ratio))
        file.write("LinearSVC\n")

def _clean_pred_result():

    cpp_filename = "cp_xnn/pred_result.txt"
    print("clean up predict file {}".format(cpp_filename))
    if os.path.isfile(cpp_filename):
        os.unlink(cpp_filename)

def gen_pred(model, rx_test, skip_ratio, loc_file):
    org_dir = os.getcwd()
    root_dir = os.path.dirname(loc_file)
    os.chdir(root_dir)

    _clean_cp_xnn()
    _save_model_to_java(model)
    _compile_java_to_executable()
    _prepare_test_data(rx_test, skip_ratio)
    _clean_pred_result()
    
    os.chdir(org_dir)