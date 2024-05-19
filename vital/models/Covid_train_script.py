# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment to disable GPU
# import glob

# from model import Model, DatasetName, load_model, remove_model

# __ORIG_WD__ = os.getcwd()
# print(__ORIG_WD__)

# os.chdir(f"{__ORIG_WD__}/../data_collectors/")
# print("++++++++++++++++++++++++")
# print(__ORIG_WD__)
# print(os.getcwd())
# from covid19_genome import Covid19Genome

# os.chdir(__ORIG_WD__)

import os
import sys

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

# Now you can import the module
from data_collectors.covid19_genome import Covid19Genome

# Remove the temporarily added path from sys.path
sys.path.remove(parent_dir)

from model import Model, DatasetName, load_model, remove_model



#########################################################
model_name = "cov19-1024e"

try:
    print("loading model")
    model = load_model(model_name)
except Exception as e:
    print (e)
    print("creating model")
    covid19_genome = Covid19Genome()
    lineages = covid19_genome.getLocalLineages(1024)
    lineages.sort()
    dataset = []
    def get_dataset():
        for lineage in lineages:
            dataset.append((lineage, covid19_genome.getLocalAccessionsPath(lineage)))
        return dataset

    portions = {
        DatasetName.trainset.name: 0.8,
        DatasetName.validset.name: 0.1,
        DatasetName.testset.name: 0.1
    }

    dataset = get_dataset()
    model = Model(model_name)
    model.create_datasets(model.get_ds_types()[0], dataset, portions)

#########################################################

sequencer_instrument_to_error_profile_map = {
    "illumina": {
        "substitution_rate": 0.005,
        "insertion_rate": 0.001,
        "deletion_rate": 0.001
    },
    "ont": {
        "substitution_rate": 0.01,
        "insertion_rate": 0.04,
        "deletion_rate": 0.04
    },
    "pacbio": {
        "substitution_rate": 0.005,
        "insertion_rate": 0.025,
        "deletion_rate": 0.025
    },
    "roche": {
        "substitution_rate": 0.002,
        "insertion_rate": 0.01,
        "deletion_rate": 0.01
    }
}
    
#########################################################

# coverage_list = [1, 4]
# ml_model_depth_list = [1, 2, 4]
coverage = 4
ml_model_depth = 1
sequencer_instrument = "illumina"
batch_size = 1024
mini_batch_size = 256

def get_model_name(ml_model_depth, coverage, sequencer_instrument):
    if not sequencer_instrument in sequencer_instrument_to_error_profile_map:
        raise Exception(f"Invalid sequencer instrument: {sequencer_instrument}")
    return f"clstm.{ml_model_depth}.{coverage}xxxx.{sequencer_instrument}"

#########################################################
def write_to_res_file(results_file, model, ml_model_name):
    print(model.ml_models[ml_model_name])
    categorical_acc = float(model.ml_models[ml_model_name].net.metrics[0].result())
    loss = float(model.ml_models[ml_model_name].net.metrics[1].result())
    results_file.write(f"\nCategorical Accuracy is {categorical_acc}\n")
    results_file.write(f"\nloss is {loss}\n")

#########################################################

# Hyperparamaters exploration
# Search Grid: Dropout, learning rate, regularizer
hp_param_exp_epochs = 1
dropout_options = [0.1, 0.2, 0.3]
lr_options = [0.01, 0.001, 0.0001]
regularizer_options = [0.00005, 0.0001, 0.0005]



#########################################################
results = open('hyperparams_explorations.txt','a')
for dropout in dropout_options:
    for lr in lr_options:
        for regularizer in regularizer_options:
            ml_model_name = get_model_name(ml_model_depth, coverage, sequencer_instrument)
            # print(ml_model_name)
            # write_to_res_file(results, model, ml_model_name)
            results.write(f"Model Name is : {ml_model_name}\n\n")
            results.write(f"Model Depth is : {ml_model_depth}\n")
            results.write(f"Coverage is : {coverage}\n")
            print(f"\n\n\nParameters are: dropout: {dropout}, learning rate: {lr}, regularizer: {regularizer}\n\n\n")
            newly_added = True
            try:
                model.add_ml_model(ml_model_name, hps={
                    "structure": model.get_ml_model_structures()[-1],
                    "d_model": model.get_ds_props()["frag_len"],
                    "seq_len": model.get_ds_props()["num_frags"],
                    "d_val": 128,
                    "d_key": 128,
                    "heads": 8,
                    "d_ff": 128+256,
                    "labels":  len(model.get_labels()),
                    "activation": "relu",
                    "optimizer": {
                        "name": "Adam",
                        "params": {
                            "learning_rate": 0.001,
                        },
                    },
                    # "encoder_repeats": ml_model_depth,
                    "LSTM_repeats": ml_model_depth,
                    "LSTM_units": 64,
                    "regularizer": {
                        "name": "l2",
                        "params": {
                            "l2": 0.0001
                        }
                    },
                    "dropout": 0.2,
                    "initializer": "he_normal"
                })
            except Exception as e:
                print (e)
                newly_added = False
                print("Model already exists")

            model.update_ds_props({"coverage": coverage,} 
            | sequencer_instrument_to_error_profile_map[sequencer_instrument])
            model.set_ds_batch_size(mini_batch_size)

            model.train(ml_model_name, epochs=hp_param_exp_epochs)
            results.write(f"Train Results : {ml_model_name}\n")
            write_to_res_file(results, model, ml_model_name)
            results.write(f"Validation Results : {ml_model_name}\n")
            model.evaluate(ml_model_name)
            write_to_res_file(results, model, ml_model_name)
            results.write(f"Test Results : {ml_model_name}\n")
            model.test(ml_model_name)
            write_to_res_file(results, model, ml_model_name)
print("DONE")

    