#!/usr/bin/env python
# coding: utf-8

# # Training covid models
# ### This notebook is an example usage of how to use the model alongside the covid-data-collector in order to train, evaluate and test the model
# #### In this notebook you will find example usages on how to use the core functionalities of the model 

# #### Import third party modules, and also the data_collector: covid19_genome and the model module

# In[3]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment to disable GPU
import glob

__ORIG_WD__ = os.getcwd()
print("++++++++++++++++++++++++")
print(__ORIG_WD__)
os.chdir(f"{__ORIG_WD__}/ProjectB---Vital/vital/models/")
from model import Model, DatasetName, load_model, remove_model


__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../data_collectors/")
print("++++++++++++++++++++++++")
print(os.getcwd())
from covid19_genome import Covid19Genome

os.chdir(__ORIG_WD__)


# #### Create a model, or try to load it, if it was already have been created.
# 
# In order to use the model, the first thing you have to do is provide it with a dataset (with the help of the data_collector). In the following cell you are provided with an example that create the dataset.
# 
# You should note that when you are creating the dataset, you are passing the dataset type. You can obtain the available dataset types in the system by calling the model class function ```get_ds_types()```

# In[2]:


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


# After you have created the model, and created its datasets. You can check which neural network structures is available. You can do that by calling the model class function ```get_ml_model_structure()```.
# 
# After you see all the ml_model structures available in the system, you can check which hyper parameters are needed to define each and every ml_model structure. This is done by calling the model class function ```get_ml_model_structure_hps()```. The ```get_ml_model_structure_hps()``` will return which hps are required, and what it their type.

# In[3]:


print(model.get_ml_model_structures())
print(model.get_ml_model_structure_hps(model.get_ml_model_structures()[-1]))


# You can also see which properties help define the current type of dataset by calling to the model class function ```get_ds_props()``` This function could be called only after the dataset have been succesfully created. This function will return the properties of the dataset as well as their values.

# In[4]:


print(model.get_ds_props())


# A use case of the system with the VitStructure model and the minhash genome datasets (a.k.a. mh_genome_ds).
# 
# In the mh_genome_ds the coverage is a dataset property that sets the genome coverage rate.
# 
# In the VitStructure, the model_depth is the number of transformer encoders.
# 
# In this example use-case these two parameters will help us define a neural network that will be trained on the dataset (with the current coverage rate)

# In[5]:


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


# In[6]:


# coverage = 1
# ml_model_depth = 1
coverage_list = [1, 4]
ml_model_depth_list = [1, 2, 4]
sequencer_instrument = "illumina"
batch_size = 1024
mini_batch_size = 256

def get_model_name(ml_model_depth, coverage, sequencer_instrument):
    if not sequencer_instrument in sequencer_instrument_to_error_profile_map:
        raise Exception(f"Invalid sequencer instrument: {sequencer_instrument}")
    return f"vit.{ml_model_depth}.{coverage}xxxx.{sequencer_instrument}"


# In[7]:


def write_to_res_file(results_file, model, ml_model_name):
    categorical_acc = float(model.ml_models[ml_model_name].net.metrics[0].result())
    loss = float(model.ml_models[ml_model_name].net.metrics[1].result())
    results_file.write(f"\nCategorical Accuracy is {categorical_acc}\n")
    results_file.write(f"\nloss is {loss}\n")


# In[8]:


results = open('results.txt','a')
for coverage in coverage_list:
    for ml_model_depth in ml_model_depth_list:
        ml_model_name = get_model_name(ml_model_depth, coverage, sequencer_instrument)
        print(ml_model_name)
        results.write(f"Model Name is : {ml_model_name}\n\n")
        results.write(f"Model Depth is : {ml_model_depth}\n")
        results.write(f"Coverage is : {coverage}\n")
        newly_added = True
        try:
            model.add_ml_model(ml_model_name, hps={
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
                "encoder_repeats": ml_model_depth,
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

        model.train(ml_model_name, epochs=30)
        results.write(f"Train Results : {ml_model_name}\n")
        write_to_res_file(results, model, model_name)
        results.write(f"Validation Results : {ml_model_name}\n")
        model.evaluate(ml_model_name)
        write_to_res_file(results, model, model_name)
        results.write(f"Test Results : {ml_model_name}\n")
        model.test(ml_model_name)
        write_to_res_file(results, model, model_name)
print("DONE")

