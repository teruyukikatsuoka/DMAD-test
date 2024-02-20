import tensorflow as tf

from .layers import *

class NN:
    def __init__(self, model):
        self.input = model.input_names
        # self.input2 = model.input_names[1]
        self.output = model.output_names[0]
        output_layers = model.output_names
        self.layers = []
        self.layers_dict = {}
        count = 0

        for layer in model.layers:
            type: str = layer.__class__.__name__
            name: str = layer.name

            if type == "Conv2D":
                self.layers.append(Conv(layer))
                self.layers_dict[name] = Conv(layer)
            elif type == "MaxPooling2D":
                self.layers.append(MaxPool(layer))
                self.layers_dict[name] = MaxPool(layer)
            elif type == "UpSampling2D":
                self.layers.append(UpSampling(layer))
                self.layers_dict[name] = UpSampling(layer)
            elif type == "AveragePooling2D":
                self.layers.append(AveragePooling2D(layer))
                self.layers_dict[name] = AveragePooling2D(layer)
            elif type == "Dense":
                self.layers.append(Dense(layer))
                self.layers_dict[name] = Dense(layer)
            # elif type == "BatchNormalization":
                # self.layers.append(BatchNorm(layer))
                # self.layers_dict[name] = BatchNorm(layer)
            # elif type == "Dropout":
            #     self.layers.append(Dropout(layer))
            #     self.layers_dict[name] = MaxPool(layer)
            elif type == "InputLayer":
                self.layers.append(Input(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Concatenate":
                self.layers.append(Concatenate(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Add":
                self.layers.append(Add(layer))
                self.layers_dict[name] = Add(layer)
            

            # elif type == "TFOpLambda":
            #      self.layers.append(TFOpLambda(layer))
            # elif type == "Conv2DTranspose":
            #     self.layers.append(ConvTranspose(layer))
            #     self.layers_dict[name] = ConvTranspose(layer)
            # elif type == "MaxPoolingWithArgmax2D":
            #     self.layers.append(MaxPoolingWithArgmax2D(layer))
            #     self.layers_dict[name] = MaxPoolingWithArgmax2D(layer)
            # elif type == "MaxUnpooling2D":
            #     self.layers.append(MaxUnpooling2D(layer))
            #     self.layers_dict[name] = MaxUnpooling2D(layer)
            # elif type == "CAM":
            #     self.layers.append(CAM(layer))
            #     self.layers_dict[name] = CAM(layer)
            # elif type == "GlobalAveragePooling2D":
            #     self.layers.append(GlobalAveragePooling2D(layer))
            #     self.layers_dict[name] = GlobalAveragePooling2D(layer)
            # elif type == "Flatten":
            #     l = Flatten(layer)
            #     self.layers.append(l)
            #     self.layers_dict[name] = l
            # elif type == "SimpleRNN":
            #     self.layers.append(SimpleRNN(layer))
            #     self.layers_dict[name] = SimpleRNN(layer)
            elif type == "Reshape":
                self.layers.append(Reshape(layer))
                self.layers_dict[name] = Reshape(layer)
            # elif type == "SamplingLayer":
            #     self.layers.append(SampligLayer(layer))
            #     self.layers_dict[name] = SampligLayer(layer)
            # elif type == "SimpleAttentionLayer":
            #     self.layers.append(SimpleAttentionLayer(layer))
            #     self.layers_dict[name] = self.layers[-1]
            else:
                assert False, "Unsupported layer type: {}".format(type)

            # TODO : Reimplenetation the case of using custom loss
            if name in output_layers:
                count += 1
            if count == len(output_layers):
                break

        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

        def get_layer_summary_with_connections(layer):
            info = {}
            connections = []
            output_index = []
            for node in layer._inbound_nodes:
                if relevant_nodes and node not in relevant_nodes:
                    continue
                for (
                    inbound_layer,
                    _,
                    tensor_index,
                    _,
                ) in node.iterate_inbound():
                    connections.append(inbound_layer.name)
                    output_index.append(tensor_index)

            info["type"] = layer.__class__.__name__
            info["parents"] = connections
            info["output_index"] = output_index
            return info

        self.connections = {}
        layers = model.layers
        for layer in layers:
            info = get_layer_summary_with_connections(layer)
            self.connections[layer.name] = info

    @tf.function
    def forward(self, inputs):
        outputs_dict = {}

        # Check inputs is required type (dict or tuple)
        assert isinstance(inputs, dict) or isinstance(inputs, tuple)

        # Assign input_si to outputs_si_dict as outputs of input layer
        if isinstance(inputs, tuple):
            outputs_dict[self.input[0]] = [inputs]
        elif isinstance(inputs, dict):
            for layer_name, input_si in inputs.items():
                outputs_dict[layer_name] = [input_si]

        for layer in self.layers:
            # Skip input layer
            if layer.name in self.input:
                continue

            connections = self.connections[layer.name]["parents"]
            index = self.connections[layer.name]["output_index"]

            if len(connections) == 1:
                # If number of input is 1, then it should be a tensor
                input_tensors = outputs_dict[connections[0]][index[0]]
            elif len(connections) > 1:
                # If number of input is more than 1, then it should be a list of tensors
                inputs_tuple = zip(connections, index)
                # input_tensors = [outputs_dict[parent][index] for parent, index in inputs_tuple]
                # これだとバグる
                input_tensors = []
                for parent, index in inputs_tuple:
                    input_tensors.append(outputs_dict[parent][index])
                
            else:
                assert False, "Invalid connections"

            # print()
            # print("--------------------")
            # print("layer.name: {}".format(layer.name))
            # if isinstance(input_tensors, list):
            #     for i in range(len(input_tensors)):
            #         print("input_tensors[{}].shape: {}".format(i, input_tensors[i].shape))
            # else:
            #     print("input_tensors.shape: {}".format(input_tensors.shape))
            # print("--------------------")
            # print()

            outputs_dict[layer.name] = layer.forward(input_tensors)

            if isinstance(outputs_dict[layer.name] ,list) == False:
                # Each output in outputs_dict must be a list of tensors
                # Even if it has only one output
                outputs_dict[layer.name] = [outputs_dict[layer.name]]

        output = outputs_dict[self.output][0]

        # 中間層の出力を確認
        # print('-----------------------------------')
        # for key, value in outputs_dict.items():
        #     print("si4dnn")
        #     print(key, value[0].numpy())
        return output, outputs_dict

    @tf.function
    def forward_si(self, input_si):
        outputs_si_dict = {}

        assert isinstance(input_si, dict) or isinstance(input_si, tuple)

        # assign input_si to outputs_si_dict as outputs of input layer
        if isinstance(input_si, tuple):
            outputs_si_dict[self.input[0]] = tuple([i] for i in input_si)
        elif isinstance(input_si, dict):
            for layer_name, input_si in input_si.items():
                outputs_si_dict[layer_name] = tuple([i] for i in input_si)

        for layer in self.layers:
            # Skip input Layer
            if layer.name in self.input:
                continue

            connections = self.connections[layer.name]["parents"]
            indexes = self.connections[layer.name]["output_index"]

            # print("-------------------------")
            # print("layer.name: {}".format(layer.name))
            # print("connections: {}".format(connections))
            # print("indexes: {}".format(indexes))
            

            if len(connections) == 1:
                # if number of layer input is 1, then x, bias, a, b should be a tensor
                x = outputs_si_dict[connections[0]][0][indexes[0]]
                bias = outputs_si_dict[connections[0]][1][indexes[0]]
                a = outputs_si_dict[connections[0]][2][indexes[0]]
                b = outputs_si_dict[connections[0]][3][indexes[0]]
                l = outputs_si_dict[connections[0]][4][indexes[0]]
                u = outputs_si_dict[connections[0]][5][indexes[0]]

            elif len(connections) > 1:
                inputs_tuple = list(zip(connections, indexes))
                # extract multiple inputs from outputs_si_dict
                # if number of layer input is more than 2, then x,bias,a,b should be a list of tensors
                x = [ outputs_si_dict[parent][0][index] for parent, index in inputs_tuple ]
                bias = [ outputs_si_dict[parent][1][index] for parent, index in inputs_tuple ]
                a = [ outputs_si_dict[parent][2][index] for parent, index in inputs_tuple ]
                b = [ outputs_si_dict[parent][3][index] for parent, index in inputs_tuple ]
                l = [ outputs_si_dict[parent][4][index] for parent, index in inputs_tuple ]
                u = [ outputs_si_dict[parent][5][index] for parent, index in inputs_tuple ]
            else:
                assert False, "Invalid connections"

            # print("input ---------------")
            # if isinstance(x, list):
            #     for i in range(len(x)):
            #         print("x[{}].shape: {}".format(i, x[i].shape))
            #         print("bias[{}].shape: {}".format(i, bias[i].shape))
            #         print("a[{}].shape: {}".format(i, a[i].shape))
            #         print("b[{}].shape: {}".format(i, b[i].shape))
            #         print("l[{}]: {}".format(i, l[i]))
            #         print("u[{}]: {}".format(i, u[i]))

            # else:
            #     print("x.shape: {}".format(x.shape))
            #     print("bias.shape: {}".format(bias.shape))
            #     print("a.shape: {}".format(a.shape))
            #     print("b.shape: {}".format(b.shape))
            #     print("l: {}".format(l))
            #     print("u: {}".format(u))

            input_si = (x, bias, a, b, l, u)
            x, bias, a, b, l, u = layer.forward_si(*input_si)
            if isinstance(x, list) == False:
                # Each output in output_si_dict must be a list
                # even if it has only one output
                x = [x]
                bias = [bias]
                a = [a]
                b = [b]
                l = [l]
                u = [u]

            # print("output ---------------")
            # if isinstance(x, list):
            #     for i in range(len(x)):
            #         print("x[{}].shape: {}".format(i, x[i].shape))
            #         print("bias[{}].shape: {}".format(i, bias[i].shape))
            #         print("a[{}].shape: {}".format(i, a[i].shape))
            #         print("b[{}].shape: {}".format(i, b[i].shape))
            #         print("l[{}].shape: {}".format(i, l[i]))
            #         print("u[{}].shape: {}".format(i, u[i]))

            # print("--------------------")
            # print()
            outputs_si_dict[layer.name] = (x, bias, a, b, l, u)

            # if l < u:
                # print("l: {}".format(l))
                # print("u: {}".format(u))
                # assert l < u

        output_x, _, _, _, l, u = outputs_si_dict[self.output]

        return l, u, output_x[0], outputs_si_dict
