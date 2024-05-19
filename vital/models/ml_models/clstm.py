import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure

class Xcept_LSTM(MLModelStructure):
    def __init__(
        self,
        hps: HPs
    ):
        super(Xcept_LSTM, self).__init__()
        self.hps = hps


        self.embedding = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
        )

        # Define the base CNN model (feature extractor)
        self.xception_layer = tf.keras.applications.Xception(weights=None, input_shape=(256, 128, 1), include_top=False)
        
        # Initialize recurrent layers list
        self.lstm_layers = []
        for _ in range(self.hps.get("LSTM_repeats")):
            self.lstm_layers.append(tf.keras.layers.LSTM(self.hps.get("LSTM_units"), return_sequences=True))  # LSTM layer with 64 units
        self.lstm_layers.append(tf.keras.layers.LSTM(self.hps.get("LSTM_units"))) # Final LSTM layer
        
        # Output layer for classification
        self.output_layer = tf.keras.layers.Dense(self.hps.get("labels"), self.hps.get("activation"))


    def call(self, inputs):
        print("Heyyyyy")
        print(f"input size: {inputs.shape}")
        x = self.embedding(inputs)
        print(f"x after embedding size: {x.shape}")
        x = tf.squeeze(x, -1)
        x = self.xception_layer(x)
        print(f"x after xception size: {x.shape}")
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))
        print(f"x before lstm size: {x.shape}")

        residual_input = x
        for i, lstm in enumerate(self.lstm_layers):
            print(f"x before lstm {i} size: {x.shape}")
            x = x + residual_input
            x = lstm(x)
            residual_input = x
        print(f"x after lstm size: {x.shape}")
        output = self.output_layer(x)
        print(f"x after dense activations size: {x.shape}")
        return output
    

    def get_config(self):
        config = super(Xcept_LSTM, self).get_config()
        config.update({
            "hps": self.hps,
        })
        return config
 
    
    def from_config(cls, config):
        config['hps'] = HPs.from_config(config['hps'])
        return cls(**config)
    

    @staticmethod
    def get_hps() -> dict:
        return {
            "d_model": "required",
            "d_val": "required",
            "d_key": "required",
            "d_ff": "required",
            "heads": "required",
            "dropout_rate": "optional",
            "regularizer": "optional",
            "initializer": "optional",
            "activation": "optional",
            "LSTM_repeats": "required",
            "labels": "required",
            "LSTM_units": "required",
        }
