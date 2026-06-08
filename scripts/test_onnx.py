import onnxruntime as ort

sess = ort.InferenceSession("results/wiktor/mlp_model.onnx")

print("Inputs:")
for x in sess.get_inputs():
    print(x.name, x.shape, x.type)

print("Outputs:")
for y in sess.get_outputs():
    print(y.name, y.shape, y.type)