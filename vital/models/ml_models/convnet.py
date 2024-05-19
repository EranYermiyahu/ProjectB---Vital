import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure

class ConvStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs,
    ):
        super(ConvStructure, self).__init__()
        self.hps = hps

        self.embedding = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
        )
        
        try:
            convclass = getattr(tf.keras.applications, self.hps.get(HPs.attributes.convclass.name))
        except:
            raise Exception(f"Convolutional class {hps.get('convclass')} not found in tf.keras.applications")

        try:
            # pascal_name = "".join([word.capitalize() for word in hps.get("convnet").split("_")])
            pascal_name =  self.hps.get(HPs.attributes.convnet.name)
            convnet = getattr(convclass, pascal_name)
            self.convnet = convnet(
                include_top=True,
                input_shape=[hps.get("seq_len"), hps.get("d_model"), 1],
                weights=None,
                pooling="avg",
                classes=hps.get("labels"),
                classifier_activation=self.hps.get("activation"),
            )
        except:
            raise Exception(f"Convolutional net {hps.get('convnet')} not found in {hps.get('convclass')}")


    def call(self, inputs):
        x = self.embedding(inputs)
        print(f"x after embedding size: {x.shape}")
        x = tf.squeeze(x, -1)
        return self.convnet(x)
    

    def get_weights(self):
        return self.convnet.get_weights()


    def set_weights(self, weights):
        self.convnet.set_weights(weights)


    @staticmethod
    def get_hps():
        return {
            "convclass": "required",
            "convnet": "required",
            "labels": "required",
            "seq_len": "required",
            "regularizer": "optional",
            "initializer": "optional",
            "d_model": "required",
            "activation": "optional",
        }