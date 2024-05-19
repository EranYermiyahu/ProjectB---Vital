import pandas as pd
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

# Hyperparamaters exploration
# Search Grid: Dropout, learning rate, regularizer
hp_param_exp_epochs = 50
    
#########################################################

coverage_list = [4, 3, 2, 1]
ml_model_depth_list = [2, 4] # Add the 8 in the end
sequencer_instrument = "illumina"
batch_size = 1024
mini_batch_size = 256

def get_model_name(ml_model_net_type, ml_model_depth, coverage, sequencer_instrument):
    if not sequencer_instrument in sequencer_instrument_to_error_profile_map:
        raise Exception(f"Invalid sequencer instrument: {sequencer_instrument}")
    return f"{ml_model_net_type}.depth{ml_model_depth}.coverage{coverage}.{sequencer_instrument}"

#########################################################
def write_to_res_file(results_file, model, ml_model_name):
    print(model.ml_models[ml_model_name])
    categorical_acc = float(model.ml_models[ml_model_name].net.metrics[0].result())
    loss = float(model.ml_models[ml_model_name].net.metrics[1].result())
    results_file.write(f"\nCategorical Accuracy is {categorical_acc}\n")
    results_file.write(f"\nloss is {loss}\n")
    return categorical_acc, loss

#########################################################

vit_hps_dict = {
    "structure": model.get_ml_model_structures()[0],
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
    "encoder_repeats": 1,
    "regularizer": {
        "name": "l2",
        "params": {
            "l2": 0.0005
        }
    },
    "dropout": 0.1,
    "initializer": "he_normal"
}
convnet_hps_dict = {
    "structure": model.get_ml_model_structures()[1],
    "d_model": model.get_ds_props()["frag_len"],
    "seq_len": model.get_ds_props()["num_frags"],
    "labels":  len(model.get_labels()),
    "activation": "relu",
    "optimizer": {
        "name": "Adam",
        "params": {
            "learning_rate": 0.001,
        },
    },
    "convnet_depth": 307, 
    "convclass": "resnet_v2",
    "convnet": "ResNet152V2",
    "regularizer": {
        "name": "l2",
        "params": {
            "l2": 0.0001
        }
    },
    "dropout": 0.2,
    "initializer": "he_normal"
}
vitEx_hps_dict = {
    "structure": model.get_ml_model_structures()[2],
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
    "encoder_repeats": 1,
    "pooling_width": 2,
    "kernel_width": 3,
    "regularizer": {
        "name": "l2",
        "params": {
            "l2": 0.0001
        }
    },
    "dropout": 0.1,
    "initializer": "he_normal"
}
clstm_hps_dict = {
    "structure": model.get_ml_model_structures()[3],
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
            "learning_rate": 0.0001,
        },
    },
    "LSTM_repeats": 1,
    "LSTM_units": 64,
    "regularizer": {
        "name": "l2",
        "params": {
            "l2": 0.00005
        }
    },
    "dropout": 0.3,
    "initializer": "he_normal"
}
# hps_list = [vit_hps_dict, convnet_hps_dict, vitEx_hps_dict, clstm_hps_dict]
# hps_list = [vit_hps_dict, vitEx_hps_dict, clstm_hps_dict]
hps_list = [clstm_hps_dict]


#########################################################
results = open('DNN_training_results.txt','a')
dataframe_list = []
for model_hps in hps_list:
    for ml_model_depth in ml_model_depth_list:
        for coverage in coverage_list:
            ml_model_net_type = model_hps["structure"].replace("Structure", "")
            if ml_model_net_type == "Conv":
                ml_model_depth = 307 
            ml_model_name = get_model_name(ml_model_net_type, ml_model_depth, coverage, sequencer_instrument)
            model_data_dict = {"Type": ml_model_net_type, "coverage": coverage, "ml_model_depth":ml_model_depth}
            results.write(f"Model Name is : {ml_model_name}\n\n")
            results.write(f"Model Depth is : {ml_model_depth}\n")
            results.write(f"Coverage is : {coverage}\n")
            newly_added = True
            try:
                if ml_model_net_type == "CLSTM":
                    key = "LSTM_repeats"
                elif ml_model_net_type == "Conv":
                    key = "convnet_depth"
                else:
                    key = "encoder_repeats"

                model_hps[key] = ml_model_depth
                model.add_ml_model(ml_model_name, hps=model_hps)
            except Exception as e:
                print (e)
                newly_added = False
                print("Model already exists")
                continue

            model.update_ds_props({"coverage": coverage,} 
            | sequencer_instrument_to_error_profile_map[sequencer_instrument])
            model.set_ds_batch_size(mini_batch_size)

            model.train(ml_model_name, epochs=hp_param_exp_epochs)
            results.write(f"Train Results : {ml_model_name}\n")
            train_categorical_acc, train_loss = write_to_res_file(results, model, ml_model_name)
            model_data_dict["train_accuracy"] = train_categorical_acc
            model_data_dict["train_loss"] = train_loss

            results.write(f"Validation Results : {ml_model_name}\n")
            model.evaluate(ml_model_name)
            val_categorical_acc, val_loss = write_to_res_file(results, model, ml_model_name)
            model_data_dict["val_accuracy"] = val_categorical_acc
            model_data_dict["val_loss"] = val_loss

            results.write(f"Test Results : {ml_model_name}\n")
            model.test(ml_model_name)
            test_categorical_acc, test_loss = write_to_res_file(results, model, ml_model_name)
            model_data_dict["test_accuracy"] = test_categorical_acc
            model_data_dict["test_loss"] = test_loss
            dataframe_list.append(model_data_dict)

            model.save_ml_model(ml_model_name)
            model.ml_models[ml_model_name].record_metric(hp_param_exp_epochs, 'test_categorical_acc', test_categorical_acc)

df = pd.DataFrame(dataframe_list)
df.to_csv('hp_explore_result.csv', index=False)   
print("DONE")

    