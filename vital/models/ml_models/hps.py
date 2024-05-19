import tensorflow as tf
import os
import json
from enum import Enum


class HPs:
    class attributes(Enum):
        structure = "structure",
        labels = "labels"
        dropout_rate = "dropout_rate",
        regularizer = "regularizer",
        initializer = "initializer",
        activation = "activation",
        optimizer = "optimizer",
        learning_rate = "learning_rate",
        loss = "loss",
        metrics = "metrics",
        batch_size = "batch_size",
        mini_batch_size = "mini_batch_size",
        LSTM_units = "LSTM_units",
        kernel_width = "kernel_width",
        pooling_width = "pooling_width",
        convclass = "convclass",
        convnet = "convnet"


    def __init__(
        self,
        hps = None,
        filepath = None
    ):
        # one of the two must be defined
        if hps is None and filepath is None:
            raise Exception("Either hps or filepath must be defined")
        
        # if filepath is defined, load the hps from the file
        if filepath is not None:
            # check that the file exists
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist")
            hps = self._load_hps(filepath)

        if HPs.attributes.structure.name in hps and HPs.attributes.labels.name in hps:
            self.hps = self.get_default()
            self.set_hps(hps)
            self.hps_get_behavior = self._get_default_get_attribute_behavior()
        else:
            raise Exception("Model hyper parameters must have \"structure\" and \"labels\" defined")
        

    def save_hps(self, dirpath):
        with open(f"{dirpath}/hps.json", "w") as f:
            f.write(json.dumps(self.hps))


    def set_hps(self, hps):
        for key in hps:
            self.hps[key] = hps[key]


    def get(self, attribute):
        return self.hps_get_behavior.get(attribute, self.default_behavior)(attribute)


    def get_dict(self):
        return self.hps


    def get_default(self):
        return {
            HPs.attributes.dropout_rate.name: 0.1, ####
            HPs.attributes.regularizer.name: { ####
                "name": "l2",
                "params": {
                    "l2": 0.01,
                }
            },
            HPs.attributes.initializer.name: "glorot_normal", ####
            HPs.attributes.optimizer.name: { ####
                "name": "AdamW",
                "params": {
                    "learning_rate": 0.001,
                }
            },
            HPs.attributes.loss.name: "categorical_crossentropy", ####
            HPs.attributes.metrics.name: ["categorical_accuracy", "AUC"], ####
            HPs.attributes.batch_size.name: 32,
            HPs.attributes.mini_batch_size.name: 32,
            HPs.attributes.activation.name: "re lu", ####
            HPs.attributes.LSTM_units.name: 64, ####
            HPs.attributes.kernel_width.name: 3, ####
            HPs.attributes.pooling_width.name: 2, ####
        }


    def get_int(self, attribute):
        if attribute in self.hps:
            return int(self.hps[attribute])
        else:
            raise Exception(f"Attribute {attribute} not found in model hyper parameters")
    
    def default_behavior(self, attribute):
            if attribute in self.hps:
                return self.hps[attribute]
            else:
                raise Exception(f"Attribute {attribute} not found in model hyper parameters")


    def _get_default_get_attribute_behavior(self):
            
        def get_from_tf_class(attribute, tf_class):
            if attribute in self.hps:
                return getattr(tf_class, self.hps[attribute]["name"])(**self.hps[attribute]["params"])
            else:
                raise Exception(f"Attribute {attribute} not found in model hyper parameters")
            

        return {
            HPs.attributes.structure.name: self.default_behavior,
            HPs.attributes.dropout_rate.name: self.default_behavior,
            HPs.attributes.regularizer.name: lambda attribute: get_from_tf_class(attribute, tf.keras.regularizers),
            HPs.attributes.initializer.name: self.default_behavior,
            HPs.attributes.optimizer.name: lambda attribute: get_from_tf_class(attribute, tf.keras.optimizers),
            HPs.attributes.loss.name: self.default_behavior,
            HPs.attributes.metrics.name: self.default_behavior,
            HPs.attributes.batch_size.name: self.get_int,
            HPs.attributes.mini_batch_size.name: self.get_int,
            HPs.attributes.activation.name: self.default_behavior,
            HPs.attributes.LSTM_units.name: self.default_behavior,
        }


    def _load_hps(self, filepath):
        with open(f"{filepath}/hps.json", "r") as f:
            return json.loads(f.read())